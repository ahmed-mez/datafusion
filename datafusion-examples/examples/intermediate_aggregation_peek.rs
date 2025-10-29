// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Example: Intermediate Aggregation Peeking (No-Grouping Queries)
//!
//! This demonstrates the intermediate aggregation peeking feature for simple
//! aggregations without GROUP BY clauses (e.g., `SELECT SUM(x), COUNT(y) FROM table`).
//!
//! # Key Features Demonstrated
//! - User-provided callback receives intermediate results as Arrow RecordBatches
//! - Results respect the aggregation's output schema
//! - Non-destructive peeking (doesn't affect final results)
//! - Works with multiple aggregation columns
//!
//! # PoC Status
//! This is a proof-of-concept implementation focused on Final aggregation mode.

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::physical_plan::{collect, displayable, ExecutionPlan};
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use datafusion::prelude::*;
use datafusion_physical_plan::aggregates::{
    peek::IntermediatePeekConfig, peek::PeekContext, AggregateExec, AggregateMode,
};
use std::sync::Arc;

/// Helper function: Recursively walks a physical plan tree and enables peek config
/// for all AggregateExec nodes found in the plan.
///
/// # Arguments
/// * `plan` - The physical plan to modify
/// * `config` - The peek configuration to apply to all AggregateExec nodes
///
/// # Returns
/// A new physical plan with peek config applied to all aggregates
fn enable_peek_for_aggregates(
    plan: Arc<dyn ExecutionPlan>,
    config: &IntermediatePeekConfig,
) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
    // Check if this node is an AggregateExec
    if let Some(agg_exec) = plan.as_any().downcast_ref::<AggregateExec>() {
        let new_agg = agg_exec
            .clone()
            .with_intermediate_peek_config(Some(config.clone()));
        return Ok(Arc::new(new_agg));
    }

    // Recursively process child nodes
    let children = plan.children();
    if children.is_empty() {
        return Ok(plan);
    }

    let new_children: datafusion::error::Result<Vec<_>> = children
        .iter()
        .map(|child| enable_peek_for_aggregates(Arc::clone(child), config))
        .collect();

    plan.with_new_children(new_children?)
}

#[tokio::main]
async fn main() -> datafusion::error::Result<()> {
    println!("=== Intermediate Aggregation Peeking Demo ===\n");

    // Create a context
    let ctx = SessionContext::new();

    // Create test data
    println!("Creating test data...");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));

    // Create batches
    let num_batches = 100;
    let rows_per_batch = 100_000;

    let mut batches = Vec::new();
    for batch_idx in 0..num_batches {
        let start = batch_idx * rows_per_batch;
        let ids: Vec<i64> = (start..start + rows_per_batch).map(|i| i as i64).collect();
        let values: Vec<i64> = (0..rows_per_batch).map(|i| (i % 1000) as i64).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(ids)),
                Arc::new(Int64Array::from(values)),
            ],
        )?;
        batches.push(batch);
    }

    println!(
        "Created {} batches with {} rows each\n",
        num_batches, rows_per_batch
    );

    // Register the table
    let provider = MemTable::try_new(schema, vec![batches])?;
    ctx.register_table("test_table", Arc::new(provider))?;

    // Create the logical plan using a query withmultiple aggregations
    let sql = "SELECT SUM(value), COUNT(value), MIN(value), MAX(value) FROM test_table";
    println!("Query: {}\n", sql);

    let dataframe = ctx.sql(sql).await?;
    let logical_plan = dataframe.logical_plan().clone();

    // Create physical plan
    let state = ctx.state();
    let planner = DefaultPhysicalPlanner::default();
    let physical_plan = planner.create_physical_plan(&logical_plan, &state).await?;

    // Display the original plan
    println!("Original Physical Plan:");
    println!("{}", displayable(physical_plan.as_ref()).indent(true));
    println!();

    // Configure intermediate peek with a custom callback
    // This callback will be called every 5ms with intermediate aggregation state
    // The results are provided as a proper Arrow RecordBatch respecting the schema
    let peek_config = IntermediatePeekConfig::new(5, |context: PeekContext| {
        // Only peek at final aggregation for this example
        if context.mode != AggregateMode::Final {
            return Ok(());
        }

        println!("=== INTERMEDIATE AGGREGATION RESULTS ===");
        println!("  Mode: {:?}", context.mode);
        println!("  Schema: {:?}", context.intermediate_batch.schema());
        println!("  Batch:");

        // Simply print the batch for this example
        if let Ok(formatted) = pretty_format_batches(&[context.intermediate_batch]) {
            println!("{}", formatted);
        }

        println!("===============================================");
        Ok(())
    });

    println!("Configuring intermediate peek:");
    println!("  Peek Config: {:?}", peek_config);
    println!();

    // Enable peeking for all aggregates in the plan
    let physical_plan_with_peek =
        enable_peek_for_aggregates(physical_plan, &peek_config)?;

    println!("Executing query with intermediate peeking enabled...");

    // Execute the plan
    let task_ctx = state.task_ctx();
    let results = collect(physical_plan_with_peek, task_ctx).await?;

    println!("\n=== FINAL RESULT ===");
    println!("{}", pretty_format_batches(&results)?);

    Ok(())
}

/*
Example output:
=== Intermediate Aggregation Peeking Demo ===

Creating test data...
Created 100 batches with 100000 rows each

Query: SELECT SUM(value), COUNT(value), MIN(value), MAX(value) FROM test_table

Original Physical Plan:
AggregateExec: mode=Final, gby=[], aggr=[sum(test_table.value), count(test_table.value), min(test_table.value), max(test_table.value)]
  CoalescePartitionsExec
    AggregateExec: mode=Partial, gby=[], aggr=[sum(test_table.value), count(test_table.value), min(test_table.value), max(test_table.value)]
      DataSourceExec: partitions=16, partition_sizes=[7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]


Configuring intermediate peek:
  Peek Config: IntermediatePeekConfig { peek_interval_ms: 5, max_groups_to_peek: Some(1000), callback: "<function>" }

Executing query with intermediate peeking enabled...
=== INTERMEDIATE AGGREGATION RESULTS ===
  Mode: Final
  Schema: Schema { fields: [Field { name: "sum(test_table.value)", data_type: Int64, nullable: true }, Field { name: "count(test_table.value)", data_type: Int64 }, Field { name: "min(test_table.value)", data_type: Int64, nullable: true }, Field { name: "max(test_table.value)", data_type: Int64, nullable: true }], metadata: {} }
  Batch:
+-----------------------+-------------------------+-----------------------+-----------------------+
| sum(test_table.value) | count(test_table.value) | min(test_table.value) | max(test_table.value) |
+-----------------------+-------------------------+-----------------------+-----------------------+
| 299700000             | 600000                  | 0                     | 999                   |
+-----------------------+-------------------------+-----------------------+-----------------------+
===============================================
=== INTERMEDIATE AGGREGATION RESULTS ===
  Mode: Final
  Schema: Schema { fields: [Field { name: "sum(test_table.value)", data_type: Int64, nullable: true }, Field { name: "count(test_table.value)", data_type: Int64 }, Field { name: "min(test_table.value)", data_type: Int64, nullable: true }, Field { name: "max(test_table.value)", data_type: Int64, nullable: true }], metadata: {} }
  Batch:
+-----------------------+-------------------------+-----------------------+-----------------------+
| sum(test_table.value) | count(test_table.value) | min(test_table.value) | max(test_table.value) |
+-----------------------+-------------------------+-----------------------+-----------------------+
| 4295700000            | 8600000                 | 0                     | 999                   |
+-----------------------+-------------------------+-----------------------+-----------------------+
===============================================

=== FINAL RESULT ===
+-----------------------+-------------------------+-----------------------+-----------------------+
| sum(test_table.value) | count(test_table.value) | min(test_table.value) | max(test_table.value) |
+-----------------------+-------------------------+-----------------------+-----------------------+
| 4995000000            | 10000000                | 0                     | 999                   |
+-----------------------+-------------------------+-----------------------+-----------------------+
*/
