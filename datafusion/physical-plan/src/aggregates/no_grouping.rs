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
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Aggregate without grouping columns

use crate::aggregates::peek::{AggregatePeek, IntermediatePeekConfig, PeekContext};
use crate::aggregates::{
    aggregate_expressions, create_accumulators, finalize_aggregation, AccumulatorItem,
    AggregateMode,
};
use crate::metrics::{BaselineMetrics, RecordOutput};
use crate::{RecordBatchStream, SendableRecordBatchStream};
use arrow::array::ArrayRef;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{internal_err, Result};
use datafusion_execution::TaskContext;
use datafusion_physical_expr::PhysicalExpr;
use futures::stream::BoxStream;
use std::borrow::Cow;
use std::sync::Arc;
use std::task::{Context, Poll};

use crate::filter::batch_filter;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use futures::stream::{Stream, StreamExt};
use itertools::Itertools;
use std::time::Instant;

use super::AggregateExec;

/// stream struct for aggregation without grouping columns
pub(crate) struct AggregateStream {
    stream: BoxStream<'static, Result<RecordBatch>>,
    schema: SchemaRef,
}

/// Actual implementation of [`AggregateStream`].
///
/// This is wrapped into yet another struct because we need to interact with the async memory management subsystem
/// during poll. To have as little code "weirdness" as possible, we chose to just use [`BoxStream`] together with
/// [`futures::stream::unfold`].
///
/// The latter requires a state object, which is [`AggregateStreamInner`].
struct AggregateStreamInner {
    schema: SchemaRef,
    mode: AggregateMode,
    input: SendableRecordBatchStream,
    baseline_metrics: BaselineMetrics,
    aggregate_expressions: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    filter_expressions: Vec<Option<Arc<dyn PhysicalExpr>>>,
    accumulators: Vec<AccumulatorItem>,
    reservation: MemoryReservation,
    finished: bool,

    /// PoC: Optional configuration for peeking at intermediate aggregation results.
    /// When set, the peek callback will be invoked periodically during execution.
    peek_config: Option<IntermediatePeekConfig>,

    /// PoC: Timestamp of the last peek operation, used to enforce peek_interval_ms
    last_peek_time: Option<Instant>,
}

impl AggregateStream {
    /// Create a new AggregateStream
    pub fn new(
        agg: &AggregateExec,
        context: Arc<TaskContext>,
        partition: usize,
    ) -> Result<Self> {
        let agg_schema = Arc::clone(&agg.schema);
        let agg_filter_expr = agg.filter_expr.clone();

        let baseline_metrics = BaselineMetrics::new(&agg.metrics, partition);
        let input = agg.input.execute(partition, Arc::clone(&context))?;

        let aggregate_expressions = aggregate_expressions(&agg.aggr_expr, &agg.mode, 0)?;
        let filter_expressions = match agg.mode {
            AggregateMode::Partial
            | AggregateMode::Single
            | AggregateMode::SinglePartitioned => agg_filter_expr,
            AggregateMode::Final | AggregateMode::FinalPartitioned => {
                vec![None; agg.aggr_expr.len()]
            }
        };
        let accumulators = create_accumulators(&agg.aggr_expr)?;

        let reservation = MemoryConsumer::new(format!("AggregateStream[{partition}]"))
            .register(context.memory_pool());

        let inner = AggregateStreamInner {
            schema: Arc::clone(&agg.schema),
            mode: agg.mode,
            input,
            baseline_metrics,
            aggregate_expressions,
            filter_expressions,
            accumulators,
            reservation,
            finished: false,
            peek_config: agg.intermediate_peek_config().cloned(),
            last_peek_time: None,
        };
        let stream = futures::stream::unfold(inner, |mut this| async move {
            if this.finished {
                return None;
            }

            loop {
                let result = match this.input.next().await {
                    Some(Ok(batch)) => {
                        // Create scope for elapsed_compute borrow
                        let timer = this.baseline_metrics.elapsed_compute().timer();
                        let result = aggregate_batch(
                            &this.mode,
                            batch,
                            &mut this.accumulators,
                            &this.aggregate_expressions,
                            &this.filter_expressions,
                        );

                        timer.done();

                        // allocate memory
                        // This happens AFTER we actually used the memory, but simplifies the whole accounting and we are OK with
                        // overshooting a bit. Also this means we either store the whole record batch or not.
                        match result
                            .and_then(|allocated| this.reservation.try_grow(allocated))
                        {
                            Ok(_) => {
                                // PoC: Peek at intermediate aggregation results if configured
                                // Now we can call the trait method without borrow conflicts
                                let _ = this.peek_intermediate_results();

                                continue;
                            }
                            Err(e) => Err(e),
                        }
                    }
                    Some(Err(e)) => Err(e),
                    None => {
                        this.finished = true;
                        let timer = this.baseline_metrics.elapsed_compute().timer();
                        let result =
                            finalize_aggregation(&mut this.accumulators, &this.mode)
                                .and_then(|columns| {
                                    RecordBatch::try_new(
                                        Arc::clone(&this.schema),
                                        columns,
                                    )
                                    .map_err(Into::into)
                                })
                                .record_output(&this.baseline_metrics);

                        timer.done();

                        result
                    }
                };

                this.finished = true;
                return Some((result, this));
            }
        });

        // seems like some consumers call this stream even after it returned `None`, so let's fuse the stream.
        let stream = stream.fuse();
        let stream = Box::pin(stream);

        Ok(Self {
            schema: agg_schema,
            stream,
        })
    }
}

impl Stream for AggregateStream {
    type Item = Result<RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        this.stream.poll_next_unpin(cx)
    }
}

impl RecordBatchStream for AggregateStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl AggregatePeek for AggregateStreamInner {
    /// Implementation of peek for no-grouping aggregations.
    ///
    /// Strategy: Uses `Accumulator::state()` to clone internal state, builds a RecordBatch,
    /// and invokes the callback. This is zero-overhead and non-destructive.
    fn peek_intermediate_results(&mut self) -> Result<()> {
        // Extract config to avoid nested borrows
        let (peek_interval_ms, callback) = match &self.peek_config {
            Some(config) => (config.peek_interval_ms, config.callback.clone()),
            None => return Ok(()),
        };

        // Check if enough time has elapsed since last peek
        let now = Instant::now();
        let should_peek = match self.last_peek_time {
            None => {
                self.last_peek_time = Some(now);
                true
            }
            Some(last) => {
                let elapsed_ms = now.duration_since(last).as_millis() as u64;
                if elapsed_ms >= peek_interval_ms {
                    self.last_peek_time = Some(now);
                    true
                } else {
                    false
                }
            }
        };

        if should_peek {
            // Build intermediate results as a RecordBatch
            match build_intermediate_batch(
                &mut self.accumulators,
                &self.mode,
                &self.schema,
            ) {
                Ok(intermediate_batch) => {
                    let context = PeekContext {
                        mode: self.mode,
                        intermediate_batch,
                    };
                    // Invoke callback - ignore errors to avoid disrupting aggregation
                    let _ = callback(context);
                }
                Err(_) => {
                    // If we can't build the batch, silently skip this peek
                }
            }
        }

        Ok(())
    }
}

/// Build an intermediate RecordBatch showing the current aggregation state
///
/// Creates a single-row RecordBatch with the current aggregate values, properly
/// formatted according to the aggregation's output schema.
///
/// # Strategy
/// For no-grouping aggregations, we use `Accumulator::state()` to get the internal
/// state and convert it to arrays. This is non-destructive as `state()` returns a
/// clone of the internal state.
///
/// # Arguments
/// * `accumulators` - Slice of accumulators to peek at
/// * `mode` - Aggregation mode (only Final/Single modes supported)
/// * `schema` - Output schema for the aggregation
///
/// # Returns
/// A single-row RecordBatch with current aggregate values, or error if unsupported mode
fn build_intermediate_batch(
    accumulators: &mut [AccumulatorItem],
    mode: &AggregateMode,
    schema: &SchemaRef,
) -> Result<RecordBatch> {
    // Only support Final/Single modes for peeking
    match mode {
        AggregateMode::Partial => {
            return internal_err!("Intermediate peeking is only supported for Final/Single aggregation modes");
        }
        AggregateMode::Final
        | AggregateMode::FinalPartitioned
        | AggregateMode::Single
        | AggregateMode::SinglePartitioned => {
            // Get state from each accumulator and convert to arrays
            let columns = accumulators
                .iter_mut()
                .map(|accumulator| {
                    accumulator.state().and_then(|state| {
                        state
                            .iter()
                            .map(|v| v.to_array())
                            .collect::<Result<Vec<ArrayRef>>>()
                    })
                })
                .flatten_ok()
                .collect::<Result<Vec<ArrayRef>>>()?;

            RecordBatch::try_new(Arc::clone(schema), columns).map_err(Into::into)
        }
    }
}

/// Perform group-by aggregation for the given [`RecordBatch`].
///
/// If successful, this returns the additional number of bytes that were allocated during this process.
///
/// TODO: Make this a member function
fn aggregate_batch(
    mode: &AggregateMode,
    batch: RecordBatch,
    accumulators: &mut [AccumulatorItem],
    expressions: &[Vec<Arc<dyn PhysicalExpr>>],
    filters: &[Option<Arc<dyn PhysicalExpr>>],
) -> Result<usize> {
    let mut allocated = 0usize;

    // 1.1 iterate accumulators and respective expressions together
    // 1.2 filter the batch if necessary
    // 1.3 evaluate expressions
    // 1.4 update / merge accumulators with the expressions' values

    // 1.1
    accumulators
        .iter_mut()
        .zip(expressions)
        .zip(filters)
        .try_for_each(|((accum, expr), filter)| {
            // 1.2
            let batch = match filter {
                Some(filter) => Cow::Owned(batch_filter(&batch, filter)?),
                None => Cow::Borrowed(&batch),
            };

            let n_rows = batch.num_rows();

            // 1.3
            let values = expr
                .iter()
                .map(|e| e.evaluate(&batch).and_then(|v| v.into_array(n_rows)))
                .collect::<Result<Vec<_>>>()?;

            // 1.4
            let size_pre = accum.size();
            let res = match mode {
                AggregateMode::Partial
                | AggregateMode::Single
                | AggregateMode::SinglePartitioned => accum.update_batch(&values),
                AggregateMode::Final | AggregateMode::FinalPartitioned => {
                    accum.merge_batch(&values)
                }
            };
            let size_post = accum.size();
            allocated += size_post.saturating_sub(size_pre);
            res
        })?;

    Ok(allocated)
}
