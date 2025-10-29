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

//! Example: Intermediate Aggregation Peeking (Grouped Queries)
//!
//! This demonstrates the intermediate aggregation peeking feature for GROUP BY queries
//! (e.g., `SELECT category, SUM(x), COUNT(*) FROM table GROUP BY category`).
//!
//! # Key Features Demonstrated
//! - Peeking at grouped aggregations as they process data
//! - Sample-based approach with configurable sample size
//! - Results include both group keys and aggregate values
//!
//! # PoC Limitation
//! **Performance Note**: For grouped aggregations, peeking uses a sample-and-reinsert
//! strategy which has overhead. The sample size is configurable via `with_max_groups()`:
//! - `Some(100)`: Low overhead, samples 100 groups
//! - `Some(1000)`: Default, moderate overhead  
//! - `None`: High overhead, peeks all groups
//!
//! For a production implementation, see `GROUPED_PEEK_LIMITATION.md` for details on
//! eliminating this overhead by extending the `GroupsAccumulator` trait.

use arrow::array::{Int64Array, RecordBatch, StringArray};
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

/// Recursively walks a physical plan tree and enables intermediate peek config
fn enable_peek_for_aggregates(
    plan: Arc<dyn ExecutionPlan>,
    config: &IntermediatePeekConfig,
) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
    // Try to downcast to AggregateExec
    if let Some(agg_exec) = plan.as_any().downcast_ref::<AggregateExec>() {
        let new_agg = agg_exec
            .clone()
            .with_intermediate_peek_config(Some(config.clone()));
        return Ok(Arc::new(new_agg));
    }

    // Recursively process children
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
    println!("=== Grouped Aggregation Peeking Demo ===\n");

    let ctx = SessionContext::new();

    // Create test data with categories
    println!("Creating test data with groups...");
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]));

    // Create multiple batches with different categories
    let num_batches = 100;
    let rows_per_batch = 50_000;

    let mut batches = Vec::new();
    for _ in 0..num_batches {
        let categories: Vec<String> = (0..rows_per_batch)
            .map(|i| format!("cat_{}", i % 10)) // 10 categories
            .collect();
        let values: Vec<i64> = (0..rows_per_batch).map(|i| (i % 1000) as i64).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(categories)),
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

    // Create a GROUP BY query
    let sql = "SELECT category, SUM(value) as total, COUNT(*) as count FROM test_table GROUP BY category";
    println!("Query: {}\n", sql);

    let dataframe = ctx.sql(sql).await?;
    let logical_plan = dataframe.logical_plan().clone();

    // Create physical plan
    let state = ctx.state();
    let planner = DefaultPhysicalPlanner::default();
    let physical_plan = planner.create_physical_plan(&logical_plan, &state).await?;

    println!("Original Physical Plan:");
    println!("{}", displayable(physical_plan.as_ref()).indent(true));
    println!();

    // Configure intermediate peek with a custom callback
    let peek_config = IntermediatePeekConfig::new(10, |context: PeekContext| {
        // Only peek at final aggregation (Final or FinalPartitioned)
        if context.mode != AggregateMode::Final
            && context.mode != AggregateMode::FinalPartitioned
        {
            return Ok(());
        }

        println!("=== INTERMEDIATE GROUPED RESULTS ===");
        println!("  Mode: {:?}", context.mode);
        println!(
            "  Total groups in peek: {}",
            context.intermediate_batch.num_rows()
        );

        // Show all groups in the peek
        if let Ok(formatted) = pretty_format_batches(&[context.intermediate_batch]) {
            let formatted_str = formatted.to_string();
            let lines: Vec<&str> = formatted_str.lines().collect();
            for line in lines.iter() {
                println!("  {}", line);
            }
        }

        println!("==========================================");
        Ok(())
    });

    println!("Configuring intermediate peek:");
    println!("  Peek Config: {:?}", peek_config);
    println!();

    // Enable peeking
    let physical_plan_with_peek =
        enable_peek_for_aggregates(physical_plan, &peek_config)?;

    println!("Executing grouped query with intermediate peeking...\n");

    // Execute the plan
    let task_ctx = state.task_ctx();
    let results = collect(physical_plan_with_peek, task_ctx).await?;

    println!("\n=== FINAL RESULT ===");
    println!("{}", pretty_format_batches(&results)?);

    Ok(())
}

/*
Example output:
=== Grouped Aggregation Peeking Demo ===

Creating test data with groups...
Created 100 batches with 50000 rows each

Query: SELECT category, SUM(value) as total, COUNT(*) as count FROM test_table GROUP BY category

Original Physical Plan:
ProjectionExec: expr=[category@0 as category, sum(test_table.value)@1 as total, count(Int64(1))@2 as count]
  AggregateExec: mode=FinalPartitioned, gby=[category@0 as category], aggr=[sum(test_table.value), count(Int64(1))]
    CoalesceBatchesExec: target_batch_size=8192
      RepartitionExec: partitioning=Hash([category@0], 16), input_partitions=16
        AggregateExec: mode=Partial, gby=[category@0 as category], aggr=[sum(test_table.value), count(Int64(1))]
          DataSourceExec: partitions=16, partition_sizes=[7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]


Configuring intermediate peek:
  Peek Config: IntermediatePeekConfig { peek_interval_ms: 10, max_groups_to_peek: Some(1000), callback: "<function>" }

Executing grouped query with intermediate peeking...

=== INTERMEDIATE GROUPED RESULTS ===
  Mode: FinalPartitioned
  Total groups in peek: 2
=== INTERMEDIATE GROUPED RESULTS ===
=== INTERMEDIATE GROUPED RESULTS ===
=== INTERMEDIATE GROUPED RESULTS ===
  Mode: FinalPartitioned
  Total groups in peek: 1
=== INTERMEDIATE GROUPED RESULTS ===
  Mode: FinalPartitioned
  Total groups in peek: 2
=== INTERMEDIATE GROUPED RESULTS ===
  Mode: FinalPartitioned
  Total groups in peek: 1
  Mode: FinalPartitioned
  Mode: FinalPartitioned
  Total groups in peek: 2
  Total groups in peek: 1
=== INTERMEDIATE GROUPED RESULTS ===
  Mode: FinalPartitioned
  Total groups in peek: 1
  +----------+-----------------------+-----------------+
  +----------+-----------------------+-----------------+
  | category | sum(test_table.value) | count(Int64(1)) |
  +----------+-----------------------+-----------------+
  | category | sum(test_table.value) | count(Int64(1)) |
  +----------+-----------------------+-----------------+
  +----------+-----------------------+-----------------+
  +----------+-----------------------+-----------------+
  | category | sum(test_table.value) | count(Int64(1)) |
  | category | sum(test_table.value) | count(Int64(1)) |
  +----------+-----------------------+-----------------+
  +----------+-----------------------+-----------------+
  | cat_6    | 250500000             | 500000          |
  +----------+-----------------------+-----------------+
  +----------+-----------------------+-----------------+
  +----------+-----------------------+-----------------+
  | cat_0    | 247500000             | 500000          |
  | category | sum(test_table.value) | count(Int64(1)) |
  | cat_9    | 252000000             | 500000          |
  +----------+-----------------------+-----------------+
  | cat_4    | 249500000             | 500000          |
  +----------+-----------------------+-----------------+
==========================================
  | cat_3    | 249000000             | 500000          |
  | cat_5    | 250000000             | 500000          |
  +----------+-----------------------+-----------------+
  | cat_7    | 251000000             | 500000          |
  +----------+-----------------------+-----------------+
  +----------+-----------------------+-----------------+
  | category | sum(test_table.value) | count(Int64(1)) |
  +----------+-----------------------+-----------------+
  | cat_1    | 248000000             | 500000          |
==========================================
==========================================
  +----------+-----------------------+-----------------+
==========================================
  +----------+-----------------------+-----------------+
  | category | sum(test_table.value) | count(Int64(1)) |
  +----------+-----------------------+-----------------+
  | cat_2    | 248500000             | 500000          |
  | cat_8    | 251500000             | 500000          |
  +----------+-----------------------+-----------------+
==========================================
==========================================
==========================================

=== FINAL RESULT ===
+----------+-----------+--------+
| category | total     | count  |
+----------+-----------+--------+
| cat_6    | 250500000 | 500000 |
| cat_4    | 249500000 | 500000 |
| cat_3    | 249000000 | 500000 |
| cat_5    | 250000000 | 500000 |
| cat_0    | 247500000 | 500000 |
| cat_9    | 252000000 | 500000 |
| cat_7    | 251000000 | 500000 |
| cat_1    | 248000000 | 500000 |
| cat_2    | 248500000 | 500000 |
| cat_8    | 251500000 | 500000 |
+----------+-----------+--------+
*/
