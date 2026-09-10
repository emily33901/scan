//! Deterministic test support for memory-consuming algorithms.

use super::{
    Access, Address, AddressRange, DirectMemoryAccess, FaultContainedMemoryAccess, MemoryError,
    MemoryMap, MemoryRegion, ModuleId, RegionKind, DEFAULT_MAX_REGIONS,
};
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FakeOperation {
    Query,
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MutationTiming {
    Before,
    After,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AccessLogEntry {
    pub operation: FakeOperation,
    /// One-based ordinal among operations of this kind.
    pub ordinal: usize,
    pub address: Address,
    /// Requested byte count (`1` for a query).
    pub len: usize,
}

#[derive(Clone, Debug)]
struct Region {
    metadata: MemoryRegion,
    bytes: Vec<u8>,
}
#[derive(Clone, Debug)]
struct Failure {
    operation: FakeOperation,
    ordinal: usize,
    error: MemoryError,
}
#[derive(Clone, Copy, Debug)]
struct Partial {
    operation: FakeOperation,
    ordinal: usize,
    bytes: usize,
}
#[derive(Clone, Debug)]
struct Mutation {
    operation: FakeOperation,
    ordinal: usize,
    timing: MutationTiming,
    address: Address,
    bytes: Vec<u8>,
}

#[derive(Debug, Default)]
struct State {
    regions: Vec<Region>,
    failures: Vec<Failure>,
    partials: Vec<Partial>,
    mutations: Vec<Mutation>,
    log: Vec<AccessLogEntry>,
    counts: [usize; 3],
}

impl State {
    fn index(op: FakeOperation) -> usize {
        match op {
            FakeOperation::Query => 0,
            FakeOperation::Read => 1,
            FakeOperation::Write => 2,
        }
    }
    fn begin(&mut self, operation: FakeOperation, address: Address, len: usize) -> usize {
        let index = Self::index(operation);
        self.counts[index] += 1;
        let ordinal = self.counts[index];
        self.log.push(AccessLogEntry {
            operation,
            ordinal,
            address,
            len,
        });
        ordinal
    }
    fn region(&self, address: Address) -> Option<&Region> {
        self.regions.iter().find(|r| r.metadata.contains(address))
    }
    fn check(&self, range: AddressRange, required: Access) -> Result<(), MemoryError> {
        let mut cursor = range.start();
        for _ in 0..DEFAULT_MAX_REGIONS {
            let region = self
                .region(cursor)
                .ok_or(MemoryError::Unmapped { address: cursor })?;
            if !region.metadata.access.contains(required) {
                return Err(MemoryError::PermissionDenied {
                    address: cursor,
                    required,
                    actual: region.metadata.access,
                });
            }
            if region.metadata.range.end() >= range.end() {
                return Ok(());
            }
            cursor = region.metadata.range.end();
        }
        Err(MemoryError::RegionLimitExceeded {
            limit: DEFAULT_MAX_REGIONS,
        })
    }
    fn copy_out(&self, address: Address, output: &mut [u8]) {
        let (mut cursor, mut done) = (address, 0);
        while done < output.len() {
            let region = self.region(cursor).expect("preflighted range");
            let offset = cursor.get() - region.metadata.range.start().get();
            let count = (region.bytes.len() - offset).min(output.len() - done);
            output[done..done + count].copy_from_slice(&region.bytes[offset..offset + count]);
            done += count;
            cursor = Address::new(cursor.get() + count);
        }
    }
    fn copy_in(&mut self, address: Address, bytes: &[u8]) {
        let (mut cursor, mut done) = (address, 0);
        while done < bytes.len() {
            let index = self
                .regions
                .iter()
                .position(|r| r.metadata.contains(cursor))
                .expect("preflighted range");
            let region = &mut self.regions[index];
            let offset = cursor.get() - region.metadata.range.start().get();
            let count = (region.bytes.len() - offset).min(bytes.len() - done);
            region.bytes[offset..offset + count].copy_from_slice(&bytes[done..done + count]);
            done += count;
            cursor = Address::new(cursor.get() + count);
        }
    }
    fn mutate(
        &mut self,
        operation: FakeOperation,
        ordinal: usize,
        timing: MutationTiming,
    ) -> Result<(), MemoryError> {
        let changes: Vec<_> = self
            .mutations
            .iter()
            .filter(|m| m.operation == operation && m.ordinal == ordinal && m.timing == timing)
            .map(|m| (m.address, m.bytes.clone()))
            .collect();
        for (address, bytes) in changes {
            self.check(address.checked_range(bytes.len())?, Access::empty())?;
            self.copy_in(address, &bytes);
        }
        Ok(())
    }
    fn failure(&self, op: FakeOperation, n: usize) -> Option<MemoryError> {
        self.failures
            .iter()
            .find(|x| x.operation == op && x.ordinal == n)
            .map(|x| x.error.clone())
    }
    fn partial(&self, op: FakeOperation, n: usize) -> Option<usize> {
        self.partials
            .iter()
            .find(|x| x.operation == op && x.ordinal == n)
            .map(|x| x.bytes)
    }
}

/// Owned synthetic memory. Clones share storage, scripts, counters and logs.
/// Synthetic addresses are never converted to host pointers.
#[derive(Clone, Debug, Default)]
pub struct FakeMemory {
    state: Arc<Mutex<State>>,
}

impl FakeMemory {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn map_region(
        &self,
        metadata: MemoryRegion,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<(), MemoryError> {
        let bytes = bytes.into();
        if metadata.range.len() != bytes.len() {
            return Err(MemoryError::InvalidValue {
                context: "fake region byte length",
            });
        }
        let mut state = self.state.lock();
        if let Some(other) = state
            .regions
            .iter()
            .find(|region| region.metadata.range.overlaps(metadata.range))
        {
            return Err(MemoryError::OverlappingRanges {
                source_range: metadata.range,
                destination: other.metadata.range,
            });
        }
        state.regions.push(Region { metadata, bytes });
        state
            .regions
            .sort_by_key(|region| region.metadata.range.start());
        Ok(())
    }
    pub fn map(
        &self,
        start: Address,
        bytes: impl Into<Vec<u8>>,
        access: Access,
        kind: RegionKind,
        module: Option<ModuleId>,
    ) -> Result<(), MemoryError> {
        let bytes = bytes.into();
        let range = start.checked_range(bytes.len())?;
        self.map_region(
            MemoryRegion {
                range,
                access,
                kind,
                module,
            },
            bytes,
        )
    }
    pub fn with_region(
        self,
        start: Address,
        bytes: impl Into<Vec<u8>>,
        access: Access,
        kind: RegionKind,
        module: Option<ModuleId>,
    ) -> Result<Self, MemoryError> {
        self.map(start, bytes, access, kind, module)?;
        Ok(self)
    }
    pub fn fail_on(
        &self,
        operation: FakeOperation,
        ordinal: usize,
        error: MemoryError,
    ) -> Result<(), MemoryError> {
        valid_ordinal(ordinal)?;
        self.state.lock().failures.push(Failure {
            operation,
            ordinal,
            error,
        });
        Ok(())
    }
    pub fn partial_on(
        &self,
        operation: FakeOperation,
        ordinal: usize,
        bytes: usize,
    ) -> Result<(), MemoryError> {
        valid_ordinal(ordinal)?;
        if operation == FakeOperation::Query {
            return Err(MemoryError::InvalidValue {
                context: "partial query operation",
            });
        }
        self.state.lock().partials.push(Partial {
            operation,
            ordinal,
            bytes,
        });
        Ok(())
    }
    pub fn mutate_on(
        &self,
        operation: FakeOperation,
        ordinal: usize,
        timing: MutationTiming,
        address: Address,
        bytes: impl Into<Vec<u8>>,
    ) -> Result<(), MemoryError> {
        valid_ordinal(ordinal)?;
        let bytes = bytes.into();
        address.checked_range(bytes.len())?;
        self.state.lock().mutations.push(Mutation {
            operation,
            ordinal,
            timing,
            address,
            bytes,
        });
        Ok(())
    }
    pub fn access_log(&self) -> Vec<AccessLogEntry> {
        self.state.lock().log.clone()
    }
    pub fn clear_access_log(&self) {
        self.state.lock().log.clear();
    }
    pub fn snapshot(&self, address: Address, len: usize) -> Result<Vec<u8>, MemoryError> {
        let range = address.checked_range(len)?;
        let state = self.state.lock();
        state.check(range, Access::empty())?;
        let mut bytes = vec![0; len];
        state.copy_out(address, &mut bytes);
        Ok(bytes)
    }
    fn read(&self, address: Address, output: &mut [u8]) -> Result<(), MemoryError> {
        let range = address.checked_range(output.len())?;
        let mut state = self.state.lock();
        state.check(range, Access::READ)?;
        let n = state.begin(FakeOperation::Read, address, output.len());
        state.mutate(FakeOperation::Read, n, MutationTiming::Before)?;
        let result = if let Some(error) = state.failure(FakeOperation::Read, n) {
            Err(error)
        } else if let Some(actual) = state.partial(FakeOperation::Read, n) {
            if actual >= output.len() {
                Err(MemoryError::InvalidValue {
                    context: "partial read byte count",
                })
            } else {
                state.copy_out(address, &mut output[..actual]);
                Err(MemoryError::PartialRead {
                    expected: output.len(),
                    actual,
                })
            }
        } else {
            state.copy_out(address, output);
            Ok(())
        };
        state.mutate(FakeOperation::Read, n, MutationTiming::After)?;
        result
    }
    fn write(&self, address: Address, bytes: &[u8]) -> Result<(), MemoryError> {
        let range = address.checked_range(bytes.len())?;
        let mut state = self.state.lock();
        state.check(range, Access::WRITE)?;
        let n = state.begin(FakeOperation::Write, address, bytes.len());
        state.mutate(FakeOperation::Write, n, MutationTiming::Before)?;
        let result = if let Some(error) = state.failure(FakeOperation::Write, n) {
            Err(error)
        } else if let Some(actual) = state.partial(FakeOperation::Write, n) {
            if actual >= bytes.len() {
                Err(MemoryError::InvalidValue {
                    context: "partial write byte count",
                })
            } else {
                state.copy_in(address, &bytes[..actual]);
                Err(MemoryError::PartialWrite {
                    expected: bytes.len(),
                    actual,
                })
            }
        } else {
            state.copy_in(address, bytes);
            Ok(())
        };
        state.mutate(FakeOperation::Write, n, MutationTiming::After)?;
        result
    }
}

fn valid_ordinal(ordinal: usize) -> Result<(), MemoryError> {
    if ordinal == 0 {
        Err(MemoryError::InvalidValue {
            context: "fake operation ordinal",
        })
    } else {
        Ok(())
    }
}

impl MemoryMap for FakeMemory {
    fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
        let mut state = self.state.lock();
        let n = state.begin(FakeOperation::Query, address, 1);
        state.mutate(FakeOperation::Query, n, MutationTiming::Before)?;
        let result = state
            .region(address)
            .map(|r| r.metadata)
            .ok_or(MemoryError::Unmapped { address })
            .and_then(|region| {
                state
                    .failure(FakeOperation::Query, n)
                    .map_or(Ok(region), Err)
            });
        state.mutate(FakeOperation::Query, n, MutationTiming::After)?;
        result
    }
}
impl DirectMemoryAccess for FakeMemory {
    unsafe fn read_bytes_into(
        &self,
        address: Address,
        output: &mut [u8],
    ) -> Result<(), MemoryError> {
        self.read(address, output)
    }
    unsafe fn write_bytes_from(&self, address: Address, bytes: &[u8]) -> Result<(), MemoryError> {
        self.write(address, bytes)
    }
}
// SAFETY: every operation is a checked copy into owned storage.
unsafe impl FaultContainedMemoryAccess for FakeMemory {
    fn try_read_bytes_into(&self, address: Address, output: &mut [u8]) -> Result<(), MemoryError> {
        super::reject_read_overlap(address, output)?;
        self.read(address, output)
    }
    unsafe fn try_write_bytes_from(
        &self,
        address: Address,
        bytes: &[u8],
    ) -> Result<(), MemoryError> {
        self.write(address, bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> FakeMemory {
        FakeMemory::new()
            .with_region(
                Address::new(0x1000),
                [1, 2, 3],
                Access::READ | Access::WRITE,
                RegionKind::Private,
                Some(ModuleId(7)),
            )
            .unwrap()
            .with_region(
                Address::new(0x1003),
                [4, 5, 6],
                Access::READ | Access::WRITE,
                RegionKind::Image,
                None,
            )
            .unwrap()
    }
    #[test]
    fn maps_metadata_and_rejects_invalid_regions() {
        let m = fixture();
        let r = m.region_containing(Address::new(0x1001)).unwrap();
        assert_eq!(r.module, Some(ModuleId(7)));
        assert_eq!(r.kind, RegionKind::Private);
        assert!(matches!(
            m.map(
                Address::new(0x1002),
                [0],
                Access::READ,
                RegionKind::Mapped,
                None
            ),
            Err(MemoryError::OverlappingRanges { .. })
        ));
        assert!(matches!(
            m.map(Address::new(9), [], Access::READ, RegionKind::Mapped, None),
            Err(MemoryError::ZeroLength)
        ));
    }
    #[test]
    fn exact_transfers_cross_adjacent_regions_through_both_traits() {
        let m = fixture();
        let mut out = [0; 4];
        unsafe { m.read_bytes_into(Address::new(0x1001), &mut out) }.unwrap();
        assert_eq!(out, [2, 3, 4, 5]);
        unsafe { m.try_write_bytes_from(Address::new(0x1002), &[9, 8, 7]) }.unwrap();
        m.try_read_bytes_into(Address::new(0x1002), &mut out[..3])
            .unwrap();
        assert_eq!(&out[..3], &[9, 8, 7]);
    }
    #[test]
    fn preflight_precedes_scripts_and_copy() {
        let m = FakeMemory::new();
        m.map(
            Address::new(10),
            [1, 2],
            Access::READ,
            RegionKind::Private,
            None,
        )
        .unwrap();
        m.fail_on(
            FakeOperation::Read,
            1,
            MemoryError::InvalidValue {
                context: "injected",
            },
        )
        .unwrap();
        let mut out = [42; 3];
        assert_eq!(
            m.try_read_bytes_into(Address::new(10), &mut out),
            Err(MemoryError::Unmapped {
                address: Address::new(12)
            })
        );
        assert_eq!(out, [42; 3]);
        assert!(m.access_log().is_empty());
    }
    #[test]
    fn partial_read_and_write_copy_only_prefix() {
        let m = fixture();
        m.partial_on(FakeOperation::Read, 1, 2).unwrap();
        m.partial_on(FakeOperation::Write, 1, 1).unwrap();
        let mut out = [0xaa; 4];
        assert_eq!(
            m.try_read_bytes_into(Address::new(0x1001), &mut out),
            Err(MemoryError::PartialRead {
                expected: 4,
                actual: 2
            })
        );
        assert_eq!(out, [2, 3, 0xaa, 0xaa]);
        assert_eq!(
            unsafe { m.try_write_bytes_from(Address::new(0x1000), &[8, 9, 10]) },
            Err(MemoryError::PartialWrite {
                expected: 3,
                actual: 1
            })
        );
        assert_eq!(m.snapshot(Address::new(0x1000), 3).unwrap(), [8, 2, 3]);
    }
    #[test]
    fn selected_failures_logs_and_ordinals_are_deterministic() {
        let m = fixture();
        m.fail_on(
            FakeOperation::Read,
            2,
            MemoryError::InvalidValue { context: "script" },
        )
        .unwrap();
        let mut b = [0];
        m.try_read_bytes_into(Address::new(0x1000), &mut b).unwrap();
        assert_eq!(
            m.try_read_bytes_into(Address::new(0x1000), &mut b),
            Err(MemoryError::InvalidValue { context: "script" })
        );
        assert_eq!(
            m.access_log().iter().map(|x| x.ordinal).collect::<Vec<_>>(),
            [1, 2]
        );

        m.fail_on(
            FakeOperation::Query,
            1,
            MemoryError::InvalidValue {
                context: "query script",
            },
        )
        .unwrap();
        assert_eq!(
            m.region_containing(Address::new(0x1000)),
            Err(MemoryError::InvalidValue {
                context: "query script"
            })
        );
    }
    #[test]
    fn mutations_run_before_and_after_an_operation() {
        let m = fixture();
        m.mutate_on(
            FakeOperation::Read,
            1,
            MutationTiming::Before,
            Address::new(0x1000),
            [9],
        )
        .unwrap();
        m.mutate_on(
            FakeOperation::Read,
            1,
            MutationTiming::After,
            Address::new(0x1000),
            [7],
        )
        .unwrap();
        let mut b = [0];
        m.try_read_bytes_into(Address::new(0x1000), &mut b).unwrap();
        assert_eq!(b, [9]);
        assert_eq!(m.snapshot(Address::new(0x1000), 1).unwrap(), [7]);
    }
    #[test]
    fn permissions_gaps_zero_lengths_and_configuration_are_checked() {
        let m = fixture();
        assert!(
            matches!(m.try_read_bytes_into(Address::new(0x1005),&mut [0;2]),Err(MemoryError::Unmapped { address }) if address==Address::new(0x1006))
        );
        let ro = FakeMemory::new();
        ro.map(
            Address::new(1),
            [0],
            Access::READ,
            RegionKind::Private,
            None,
        )
        .unwrap();
        assert!(matches!(
            unsafe { ro.write_bytes_from(Address::new(1), &[2]) },
            Err(MemoryError::PermissionDenied { .. })
        ));
        assert!(matches!(
            m.try_read_bytes_into(Address::new(0x1000), &mut []),
            Err(MemoryError::ZeroLength)
        ));
        assert!(matches!(
            m.partial_on(FakeOperation::Query, 1, 0),
            Err(MemoryError::InvalidValue { .. })
        ));
        assert!(matches!(
            m.fail_on(FakeOperation::Read, 0, MemoryError::ZeroLength),
            Err(MemoryError::InvalidValue { .. })
        ));
    }
    #[test]
    fn clones_share_thread_safe_state() {
        let m = fixture();
        let clone = m.clone();
        std::thread::spawn(move || {
            unsafe { clone.try_write_bytes_from(Address::new(0x1000), &[42]) }.unwrap()
        })
        .join()
        .unwrap();
        assert_eq!(m.snapshot(Address::new(0x1000), 1).unwrap(), [42]);
    }

    #[test]
    fn transfer_preflight_observes_the_default_region_limit() {
        let m = FakeMemory::new();
        for offset in 0..=DEFAULT_MAX_REGIONS {
            m.map(
                Address::new(0x2000 + offset),
                [0],
                Access::READ,
                RegionKind::Private,
                None,
            )
            .unwrap();
        }
        let mut output = vec![1; DEFAULT_MAX_REGIONS + 1];
        assert_eq!(
            m.try_read_bytes_into(Address::new(0x2000), &mut output),
            Err(MemoryError::RegionLimitExceeded {
                limit: DEFAULT_MAX_REGIONS
            })
        );
        assert!(output.iter().all(|byte| *byte == 1));
    }
}
