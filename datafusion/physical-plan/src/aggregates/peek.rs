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

//! Intermediate aggregation peeking support (Proof of Concept)
//!
//! This module provides infrastructure for observing intermediate aggregation results
//! during query execution. This is useful for monitoring long-running aggregations
//! and understanding query progress.
//!
//! # Overview
//!
//! The peeking feature allows users to provide a callback function that will be invoked
//! periodically during aggregation execution with intermediate results formatted as
//! Arrow RecordBatches.
//!
//! # Example
//!
//! ```ignore
//! use datafusion_physical_plan::aggregates::{IntermediatePeekConfig, PeekContext};
//!
//! let config = IntermediatePeekConfig::new(1000, |context: PeekContext| {
//!     println!("Progress: {} rows", context.intermediate_batch.num_rows());
//!     Ok(())
//! });
//!
//! let agg_exec = agg_exec.with_intermediate_peek_config(Some(config));
//! ```

use arrow::record_batch::RecordBatch;
use datafusion_common::Result;
use std::sync::Arc;

use super::AggregateMode;

/// Context provided to the peek callback containing intermediate aggregation state.
///
/// The intermediate results are provided as an Arrow RecordBatch that respects
/// the aggregation's output schema.
#[derive(Debug, Clone)]
pub struct PeekContext {
    /// The aggregation mode being executed
    pub mode: AggregateMode,

    /// Intermediate aggregation results as a RecordBatch
    /// - For no-grouping aggregations: Single-row batch with current aggregate values
    /// - For grouped aggregations: One row per group (may be sampled based on config)
    pub intermediate_batch: RecordBatch,
}

/// Type for the callback function that handles intermediate aggregation results.
///
/// The callback receives a `PeekContext` containing the intermediate results
/// as a properly formatted RecordBatch with the correct schema.
///
/// # Arguments
/// - `PeekContext`: Contains the aggregation mode and intermediate results
///
/// # Returns
/// - `Ok(())` on success
/// - `Err(...)` on error (errors are currently ignored by implementations)
pub type PeekCallback = Arc<dyn Fn(PeekContext) -> Result<()> + Send + Sync>;

/// Configuration for peeking at intermediate aggregation results during execution.
///
/// This is a proof-of-concept feature that enables observing aggregation progress
/// by invoking a user-provided callback with intermediate results at regular intervals.
///
/// # Example
///
/// ```ignore
/// use datafusion_physical_plan::aggregates::{IntermediatePeekConfig, PeekContext};
///
/// let config = IntermediatePeekConfig::new(1000, |context: PeekContext| {
///     println!("Progress: {} rows", context.intermediate_batch.num_rows());
///     Ok(())
/// });
/// ```
pub struct IntermediatePeekConfig {
    /// Minimum duration between peeks in milliseconds
    pub peek_interval_ms: u64,

    /// Callback function to handle intermediate results
    pub callback: PeekCallback,

    /// Maximum number of groups to sample for grouped aggregations (GROUP BY queries).
    /// - `Some(n)`: Sample up to n groups (reduces re-insertion overhead)
    /// - `None`: Peek all groups (higher overhead)
    ///
    /// Note: Only applies to grouped aggregations. No-grouping aggregations always
    /// peek the single aggregate row with no overhead.
    pub max_groups_to_peek: Option<usize>,
}

impl IntermediatePeekConfig {
    /// Create a new peek configuration with a custom callback.
    ///
    /// # Arguments
    /// * `peek_interval_ms` - Minimum duration between peeks in milliseconds
    /// * `callback` - Function to handle intermediate results. Returns Result to allow error handling.
    ///
    /// # Default Configuration
    /// - Samples up to 1000 groups for grouped aggregations (balances overhead vs visibility)
    /// - Change via [`Self::with_max_groups`] if needed
    ///
    /// # Example
    ///
    /// ```ignore
    /// use arrow::util::pretty::pretty_format_batches;
    ///
    /// let config = IntermediatePeekConfig::new(1000, |context| {
    ///     if let Ok(formatted) = pretty_format_batches(&[context.intermediate_batch]) {
    ///         println!("{}", formatted);
    ///     }
    ///     Ok(())
    /// });
    /// ```
    pub fn new<F>(peek_interval_ms: u64, callback: F) -> Self
    where
        F: Fn(PeekContext) -> Result<()> + Send + Sync + 'static,
    {
        Self {
            peek_interval_ms,
            callback: Arc::new(callback),
            max_groups_to_peek: Some(1000),
        }
    }

    /// Set the maximum number of groups to peek for grouped aggregations (GROUP BY).
    ///
    /// # Arguments
    /// * `max_groups` - Maximum groups to sample per peek
    ///   - `Some(n)`: Sample up to n groups (recommended: 100-1000)
    ///   - `None`: Peek all groups (use with caution for queries with many groups)
    ///
    /// # Performance Impact
    ///
    /// For grouped aggregations, peeking requires emit + re-insertion.
    /// Overhead is proportional to `max_groups`.
    ///
    /// The current implementation is a proof of concept and will be improved in the future.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Sample only 100 groups for low overhead
    /// let config = IntermediatePeekConfig::new(1000, callback)
    ///     .with_max_groups(Some(100));
    ///
    /// // Peek all groups (higher overhead)
    /// let config = IntermediatePeekConfig::new(1000, callback)
    ///     .with_max_groups(None);
    /// ```
    pub fn with_max_groups(mut self, max_groups: Option<usize>) -> Self {
        self.max_groups_to_peek = max_groups;
        self
    }
}

impl std::fmt::Debug for IntermediatePeekConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntermediatePeekConfig")
            .field("peek_interval_ms", &self.peek_interval_ms)
            .field("max_groups_to_peek", &self.max_groups_to_peek)
            .field("callback", &"<function>")
            .finish()
    }
}

impl Clone for IntermediatePeekConfig {
    fn clone(&self) -> Self {
        Self {
            peek_interval_ms: self.peek_interval_ms,
            max_groups_to_peek: self.max_groups_to_peek,
            callback: Arc::clone(&self.callback),
        }
    }
}

/// Trait for peeking at intermediate aggregation results during execution.
///
/// This provides a unified interface for different aggregation stream implementations
/// to support intermediate result peeking. Each implementation handles peeking
/// according to its specific data structures and performance characteristics.
///
/// # Implementations
///
/// - **`AggregateStreamInner`** (no-grouping): Zero-overhead, clones accumulator state
/// - **`GroupedHashAggregateStream`** (GROUP BY): Sample-and-reinsert strategy with overhead
/// - **`GroupedTopKAggregateStream`** (TopK): Returns `not_impl_err!`
///
/// # Usage
///
/// The trait provides a consistent API, but implementations may vary in overhead and strategy.
/// See individual implementation documentation for details.
///
/// # Example Implementation
///
/// ```ignore
/// impl AggregatePeek for MyAggregateStream {
///     fn peek_intermediate_results(&mut self) -> Result<()> {
///         // 1. Check if configured
///         let config = self.peek_config.as_ref()?;
///         
///         // 2. Check timing
///         if !should_peek_now(...) {
///             return Ok(());
///         }
///         
///         // 3. Build RecordBatch with intermediate state
///         let batch = self.build_peek_batch()?;
///         
///         // 4. Invoke callback
///         (config.callback)(PeekContext { mode: self.mode, intermediate_batch: batch })
///     }
/// }
/// ```
pub(crate) trait AggregatePeek {
    /// Peek at intermediate aggregation results if configured and timing allows.
    ///
    /// This method should:
    /// 1. Check if peeking is configured (return `Ok(())` if not)
    /// 2. Check if enough time has elapsed since last peek
    /// 3. Build a RecordBatch with current intermediate results
    /// 4. Invoke the callback with the batch
    ///
    /// # Non-Destructive Requirement
    ///
    /// Implementations **must** be non-destructive - the aggregation state should
    /// remain unchanged (or be restored) after peeking. Final query results must
    /// not be affected by peeking.
    ///
    /// # Returns
    /// - `Ok(())` on success or if peeking is not configured
    /// - `Err(...)` if peeking is not supported for this aggregation type
    fn peek_intermediate_results(&mut self) -> Result<()>;
}
