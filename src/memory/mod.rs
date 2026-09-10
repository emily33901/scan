//! Checked address arithmetic and process-memory access contracts.
//!
//! [`MemoryMap`] performs bounded, complete-range validation. That validation
//! is deliberately separate from copying: mappings are snapshots and cannot
//! prove allocation lifetime, initialization, or synchronization.

use std::fmt;
use std::marker::PhantomData;
use std::mem::{align_of, size_of, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::rc::Rc;

use bitflags::bitflags;
use thiserror::Error;

/// An address in the current process.
///
/// Zero is representable so that copied pointer values can be inspected before
/// an operation explicitly requires them to be non-null.
#[derive(Clone, Copy, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct Address(usize);

impl Address {
    pub const NULL: Self = Self(0);

    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    pub const fn get(self) -> usize {
        self.0
    }

    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    pub fn require_non_null(self) -> Result<Self, MemoryError> {
        if self.is_null() {
            Err(MemoryError::NullAddress)
        } else {
            Ok(self)
        }
    }

    pub fn checked_add(self, offset: usize) -> Result<Self, MemoryError> {
        self.0
            .checked_add(offset)
            .map(Self)
            .ok_or(MemoryError::AddressOverflow)
    }

    pub fn checked_sub(self, offset: usize) -> Result<Self, MemoryError> {
        self.0
            .checked_sub(offset)
            .map(Self)
            .ok_or(MemoryError::AddressOverflow)
    }

    pub fn checked_add_signed(self, offset: i64) -> Result<Self, MemoryError> {
        let magnitude =
            usize::try_from(offset.unsigned_abs()).map_err(|_| MemoryError::AddressOverflow)?;
        let value = if offset >= 0 {
            self.0.checked_add(magnitude)
        } else {
            self.0.checked_sub(magnitude)
        };
        value.map(Self).ok_or(MemoryError::AddressOverflow)
    }

    pub fn checked_range(self, len: usize) -> Result<AddressRange, MemoryError> {
        AddressRange::new(self, len)
    }

    pub fn is_aligned_to(self, alignment: usize) -> Result<bool, MemoryError> {
        validate_alignment(alignment)?;
        Ok(self.0 & (alignment - 1) == 0)
    }

    pub fn require_alignment(self, alignment: usize) -> Result<Self, MemoryError> {
        if self.is_aligned_to(alignment)? {
            Ok(self)
        } else {
            Err(MemoryError::Misaligned {
                address: self,
                alignment,
            })
        }
    }
}

impl From<usize> for Address {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<Address> for usize {
    fn from(value: Address) -> Self {
        value.0
    }
}

impl fmt::Debug for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Address({self:#x})")
    }
}

impl fmt::Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#x}", self.0)
    }
}

impl fmt::LowerHex for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}

/// A non-empty half-open address range.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct AddressRange {
    start: Address,
    end: Address,
}

impl AddressRange {
    pub fn new(start: Address, len: usize) -> Result<Self, MemoryError> {
        if len == 0 {
            return Err(MemoryError::ZeroLength);
        }
        let end = start.checked_add(len)?;
        Ok(Self { start, end })
    }

    pub fn from_bounds(start: Address, end: Address) -> Result<Self, MemoryError> {
        if end <= start {
            return if end == start {
                Err(MemoryError::ZeroLength)
            } else {
                Err(MemoryError::AddressOverflow)
            };
        }
        Ok(Self { start, end })
    }

    pub const fn start(self) -> Address {
        self.start
    }

    pub const fn end(self) -> Address {
        self.end
    }

    pub const fn len(self) -> usize {
        self.end.0 - self.start.0
    }

    pub const fn is_empty(self) -> bool {
        false
    }

    pub const fn contains(self, address: Address) -> bool {
        self.start.0 <= address.0 && address.0 < self.end.0
    }

    pub const fn contains_range(self, other: Self) -> bool {
        self.start.0 <= other.start.0 && other.end.0 <= self.end.0
    }

    pub const fn overlaps(self, other: Self) -> bool {
        self.start.0 < other.end.0 && other.start.0 < self.end.0
    }
}

bitflags! {
    /// Permissions required from or reported for a memory region.
    #[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
    pub struct Access: u8 {
        const READ = 1 << 0;
        const WRITE = 1 << 1;
        const EXECUTE = 1 << 2;
    }
}

/// Platform-neutral classification of a memory mapping.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum RegionKind {
    Private,
    Image,
    Mapped,
    Shared,
    Unknown,
}

/// Stable identity assigned to a discovered module snapshot.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(transparent)]
pub struct ModuleId(pub u64);

/// A single contiguous memory mapping returned by a backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryRegion {
    pub range: AddressRange,
    pub access: Access,
    pub kind: RegionKind,
    pub module: Option<ModuleId>,
}

impl MemoryRegion {
    pub const fn contains(self, address: Address) -> bool {
        self.range.contains(address)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlatformOperation {
    Query,
    Read,
    Write,
}

/// Failures produced by checked address and memory operations.
#[derive(Debug, Error, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum MemoryError {
    #[error("null address is not valid for this operation")]
    NullAddress,
    #[error("address arithmetic overflow or underflow")]
    AddressOverflow,
    #[error("zero-length memory operations are not supported")]
    ZeroLength,
    #[error("alignment {alignment} is not a nonzero power of two")]
    InvalidAlignment { alignment: usize },
    #[error("address {address} is not aligned to {alignment}")]
    Misaligned { address: Address, alignment: usize },
    #[error("requested length {requested} exceeds limit {limit}")]
    LengthExceeded { requested: usize, limit: usize },
    #[error("range traversal exceeded its {limit}-region limit")]
    RegionLimitExceeded { limit: usize },
    #[error("address {address} is not mapped")]
    Unmapped { address: Address },
    #[error("range at {address} requires {required:?}, but region has {actual:?}")]
    PermissionDenied {
        address: Address,
        required: Access,
        actual: Access,
    },
    #[error("mapping query for {address} did not return a containing, advancing region")]
    NonProgressingRegion { address: Address },
    #[error("read transferred {actual} of {expected} bytes")]
    PartialRead { expected: usize, actual: usize },
    #[error("write transferred {actual} of {expected} bytes")]
    PartialWrite { expected: usize, actual: usize },
    #[error("source range {source_range:?} overlaps destination range {destination:?}")]
    OverlappingRanges {
        source_range: AddressRange,
        destination: AddressRange,
    },
    #[error("string was not terminated within {max_len} bytes")]
    UnterminatedString { max_len: usize },
    #[error("invalid value for {context}")]
    InvalidValue { context: &'static str },
    #[error("{operation:?} is unsupported on this platform")]
    UnsupportedPlatform { operation: PlatformOperation },
    #[error("platform {operation:?} failed with code {code}")]
    Platform {
        operation: PlatformOperation,
        code: i64,
    },
}

pub fn validate_alignment(alignment: usize) -> Result<(), MemoryError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        Err(MemoryError::InvalidAlignment { alignment })
    } else {
        Ok(())
    }
}

fn reject_read_overlap(address: Address, output: &mut [u8]) -> Result<(), MemoryError> {
    if output.is_empty() {
        return Err(MemoryError::ZeroLength);
    }
    let source_range = address.checked_range(output.len())?;
    let destination = Address::new(output.as_mut_ptr() as usize).checked_range(output.len())?;
    if source_range.overlaps(destination) {
        Err(MemoryError::OverlappingRanges {
            source_range,
            destination,
        })
    } else {
        Ok(())
    }
}

/// Supplies point-in-time mapping information for current-process addresses.
pub trait MemoryMap: Send + Sync {
    fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError>;

    fn check_range(
        &self,
        range: AddressRange,
        required: Access,
        max_regions: usize,
    ) -> Result<(), MemoryError> {
        if max_regions == 0 {
            return Err(MemoryError::RegionLimitExceeded { limit: 0 });
        }

        let mut cursor = range.start();
        for _ in 0..max_regions {
            let region = self.region_containing(cursor)?;
            if !region.contains(cursor) || region.range.end() <= cursor {
                return Err(MemoryError::NonProgressingRegion { address: cursor });
            }
            if !region.access.contains(required) {
                return Err(MemoryError::PermissionDenied {
                    address: cursor,
                    required,
                    actual: region.access,
                });
            }
            if region.range.end() >= range.end() {
                return Ok(());
            }
            cursor = region.range.end();
        }

        Err(MemoryError::RegionLimitExceeded { limit: max_regions })
    }
}

/// Direct, in-process byte copying after complete-range preflight.
///
/// # Safety
///
/// Implementations must not construct references from `address`. Callers must
/// keep the full range mapped, permitted, and initialized where read from the
/// beginning of preflight through completion of the copy, and must uphold all
/// raw-pointer synchronization requirements.
pub trait DirectMemoryAccess: MemoryMap {
    /// Copies exactly `output.len()` bytes into `output` after preflight.
    ///
    /// # Safety
    ///
    /// The full source range must remain mapped, readable, initialized, valid
    /// for raw copying, and properly synchronized through the operation.
    unsafe fn read_bytes_into(
        &self,
        address: Address,
        output: &mut [u8],
    ) -> Result<(), MemoryError>;

    /// Copies exactly `bytes.len()` bytes to `address` after preflight.
    ///
    /// # Safety
    ///
    /// The full destination range must remain mapped, writable, valid for raw
    /// copying, and properly synchronized. Replacing the bytes must preserve
    /// all foreign program invariants.
    unsafe fn write_bytes_from(&self, address: Address, bytes: &[u8]) -> Result<(), MemoryError>;
}

/// A memory backend whose read implementation contains invalid-address faults.
///
/// # Safety
///
/// An implementation must accept arbitrary address/length inputs without
/// causing Rust undefined behavior or a native memory fault. `Ok(())` may be
/// returned only after an exact transfer. Reads are safe; writes remain unsafe
/// because mapping checks cannot establish foreign program invariants.
pub unsafe trait FaultContainedMemoryAccess: MemoryMap {
    fn try_read_bytes_into(&self, address: Address, output: &mut [u8]) -> Result<(), MemoryError>;

    /// Attempts an exact fault-contained write.
    ///
    /// # Safety
    ///
    /// Replacing the destination bytes must preserve all foreign program
    /// invariants even though invalid-address faults are contained.
    unsafe fn try_write_bytes_from(
        &self,
        address: Address,
        bytes: &[u8],
    ) -> Result<(), MemoryError>;
}

/// Default bound for a complete-range mapping walk.
pub const DEFAULT_MAX_REGIONS: usize = 1_024;

/// Typed operations built on a direct-copy backend.
pub trait DirectMemoryAccessExt: DirectMemoryAccess {
    /// Copies a value from a foreign address.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DirectMemoryAccess::read_bytes_into`] for the
    /// complete value range. `T: AnyBitPattern` guarantees that an exact byte
    /// transfer is a valid `T`; it does not establish source lifetime.
    unsafe fn read_value<T: bytemuck::AnyBitPattern>(
        &self,
        address: Address,
    ) -> Result<T, MemoryError> {
        let mut value = MaybeUninit::<T>::uninit();
        let len = size_of::<T>();
        if len == 0 {
            return Err(MemoryError::ZeroLength);
        }
        // SAFETY: MaybeUninit<T> may be viewed as writable bytes. T is only
        // constructed after the backend reports an exact successful transfer.
        let output =
            unsafe { std::slice::from_raw_parts_mut(value.as_mut_ptr().cast::<u8>(), len) };
        // SAFETY: delegated to this method's caller.
        unsafe { self.read_bytes_into(address, output)? };
        // SAFETY: the backend's contract promises exact initialization on Ok,
        // and AnyBitPattern makes every initialized bit pattern valid for T.
        Ok(unsafe { value.assume_init() })
    }

    /// Copies the initialized representation of a value to a foreign address.
    ///
    /// # Safety
    ///
    /// The caller must uphold [`DirectMemoryAccess::write_bytes_from`] and
    /// guarantee that replacing the destination bytes preserves all foreign
    /// program invariants.
    unsafe fn write_value<T: bytemuck::NoUninit>(
        &self,
        address: Address,
        value: &T,
    ) -> Result<(), MemoryError> {
        let bytes = bytemuck::bytes_of(value);
        if bytes.is_empty() {
            return Err(MemoryError::ZeroLength);
        }
        // SAFETY: delegated to this method's caller.
        unsafe { self.write_bytes_from(address, bytes) }
    }
}

impl<M: DirectMemoryAccess + ?Sized> DirectMemoryAccessExt for M {}

/// Typed operations built on a fault-contained backend.
pub trait FaultContainedMemoryAccessExt: FaultContainedMemoryAccess {
    fn try_read_value<T: bytemuck::AnyBitPattern>(
        &self,
        address: Address,
    ) -> Result<T, MemoryError> {
        let mut value = MaybeUninit::<T>::uninit();
        let len = size_of::<T>();
        if len == 0 {
            return Err(MemoryError::ZeroLength);
        }
        // SAFETY: MaybeUninit<T> is valid writable byte storage. The value is
        // constructed only after the fault-contained backend reports success.
        let output =
            unsafe { std::slice::from_raw_parts_mut(value.as_mut_ptr().cast::<u8>(), len) };
        self.try_read_bytes_into(address, output)?;
        // SAFETY: exact initialization is guaranteed on Ok and T accepts every
        // initialized bit pattern.
        Ok(unsafe { value.assume_init() })
    }

    /// # Safety
    ///
    /// The caller must guarantee that replacing the destination bytes
    /// preserves all foreign program invariants.
    unsafe fn try_write_value<T: bytemuck::NoUninit>(
        &self,
        address: Address,
        value: &T,
    ) -> Result<(), MemoryError> {
        let bytes = bytemuck::bytes_of(value);
        if bytes.is_empty() {
            return Err(MemoryError::ZeroLength);
        }
        // SAFETY: delegated to this method's caller.
        unsafe { self.try_write_bytes_from(address, bytes) }
    }
}

impl<M: FaultContainedMemoryAccess + ?Sized> FaultContainedMemoryAccessExt for M {}

/// The production memory backend for the current process.
#[derive(Clone, Copy, Debug, Default)]
pub struct ProcessMemory {
    _private: (),
}

impl ProcessMemory {
    pub const fn new() -> Self {
        Self { _private: () }
    }

    /// Establishes the stability contract for checked direct operations once.
    ///
    /// # Safety
    ///
    /// Every range used by `operation`, including dynamically derived ranges,
    /// must remain allocated, mapped, permitted, initialized where read, valid
    /// for raw access, and correctly synchronized from its final preflight
    /// through its copy. Values borrowing the fresh scope cannot escape.
    pub unsafe fn with_scope<R>(
        &self,
        operation: impl for<'scope> FnOnce(&mut DirectMemoryScope<'scope, Self>) -> R,
    ) -> R {
        // SAFETY: delegated to this method's caller.
        unsafe { with_direct_memory_scope(self, operation) }
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
impl MemoryMap for ProcessMemory {
    fn region_containing(&self, _address: Address) -> Result<MemoryRegion, MemoryError> {
        Err(MemoryError::UnsupportedPlatform {
            operation: PlatformOperation::Query,
        })
    }
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
impl DirectMemoryAccess for ProcessMemory {
    unsafe fn read_bytes_into(
        &self,
        _address: Address,
        _output: &mut [u8],
    ) -> Result<(), MemoryError> {
        Err(MemoryError::UnsupportedPlatform {
            operation: PlatformOperation::Read,
        })
    }

    unsafe fn write_bytes_from(&self, _address: Address, _bytes: &[u8]) -> Result<(), MemoryError> {
        Err(MemoryError::UnsupportedPlatform {
            operation: PlatformOperation::Write,
        })
    }
}

/// Runs a direct operation in a fresh, invariant, non-thread-transferable scope.
///
/// # Safety
///
/// The caller must uphold the same scope-wide contract documented by
/// this function for `memory` and every access in `operation`: all copied
/// ranges must remain allocated, mapped, permitted, initialized where read,
/// valid for raw access, and properly synchronized through each copy.
pub unsafe fn with_direct_memory_scope<M, R>(
    memory: &M,
    operation: impl for<'scope> FnOnce(&mut DirectMemoryScope<'scope, M>) -> R,
) -> R
where
    M: DirectMemoryAccess + ?Sized,
{
    let mut scope = DirectMemoryScope {
        memory,
        _brand: PhantomData,
        _not_send_sync: PhantomData,
    };
    operation(&mut scope)
}

/// A non-escaping capability for checked direct access.
pub struct DirectMemoryScope<'scope, M: ?Sized = ProcessMemory> {
    memory: &'scope M,
    _brand: PhantomData<fn(&'scope ()) -> &'scope ()>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl<'scope, M: DirectMemoryAccess + ?Sized> DirectMemoryScope<'scope, M> {
    pub const fn memory(&self) -> &M {
        self.memory
    }

    pub fn check_range(
        &self,
        range: AddressRange,
        required: Access,
        max_regions: usize,
    ) -> Result<(), MemoryError> {
        self.memory.check_range(range, required, max_regions)
    }

    /// Copies bytes under the contract established when this scope was made.
    pub fn read_bytes_into(&self, address: Address, output: &mut [u8]) -> Result<(), MemoryError> {
        reject_read_overlap(address, output)?;
        // SAFETY: creation of the scope established the backend preconditions;
        // overlap with the Rust output borrow was rejected above.
        unsafe { self.memory.read_bytes_into(address, output) }
    }

    pub fn read_value<T: bytemuck::AnyBitPattern>(
        &self,
        address: Address,
    ) -> Result<T, MemoryError> {
        let mut value = MaybeUninit::<T>::uninit();
        let len = size_of::<T>();
        if len == 0 {
            return Err(MemoryError::ZeroLength);
        }
        // SAFETY: MaybeUninit<T> is writable byte storage.
        let output =
            unsafe { std::slice::from_raw_parts_mut(value.as_mut_ptr().cast::<u8>(), len) };
        self.read_bytes_into(address, output)?;
        // SAFETY: the exact successful copy initialized every byte and T accepts
        // every initialized bit pattern.
        Ok(unsafe { value.assume_init() })
    }

    /// # Safety
    ///
    /// Replacing these foreign bytes must preserve host program invariants.
    pub unsafe fn write_bytes_from(
        &mut self,
        address: Address,
        bytes: &[u8],
    ) -> Result<(), MemoryError> {
        if bytes.is_empty() {
            return Err(MemoryError::ZeroLength);
        }
        // SAFETY: scope creation established raw-access validity; semantic
        // validity is delegated to this method's caller.
        unsafe { self.memory.write_bytes_from(address, bytes) }
    }

    /// Acquires a shared Rust-reference capability for foreign storage.
    ///
    /// # Safety
    ///
    /// In addition to the checked mapping, non-null, alignment, and size
    /// requirements, `address` must have valid provenance for one live,
    /// initialized `T` allocation for `'a`. Its size must be at most
    /// `isize::MAX`; it must remain readable and immovable, and all mutation,
    /// aliasing, data-race, destruction, reprotection, and unload rules of an
    /// ordinary `&T` must hold for the complete lifetime of the result.
    pub unsafe fn foreign_ref<'a, T: Sized>(
        &'a self,
        address: Address,
    ) -> Result<ForeignRef<'a, T>, ForeignReferenceError> {
        validate_foreign::<T>(self.memory, address, Access::READ)?;
        // SAFETY: the caller promises valid exposed provenance and a live T.
        let pointer = unsafe { NonNull::new_unchecked(address.get() as *mut T) };
        Ok(ForeignRef {
            pointer,
            _lifetime: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    /// Acquires an exclusive Rust-reference capability for foreign storage.
    ///
    /// # Safety
    ///
    /// All requirements of [`Self::foreign_ref`] apply, and this capability
    /// must have exclusive access for `'a`. Safe code may replace, move, or
    /// drop the value through `DerefMut`; each such operation must be valid for
    /// the host object. Use a copied field write if that cannot be guaranteed.
    pub unsafe fn foreign_mut<'a, T: Sized>(
        &'a mut self,
        address: Address,
    ) -> Result<ForeignMut<'a, T>, ForeignReferenceError> {
        validate_foreign::<T>(self.memory, address, Access::READ | Access::WRITE)?;
        // SAFETY: the caller promises valid exposed provenance and unique T.
        let pointer = unsafe { NonNull::new_unchecked(address.get() as *mut T) };
        Ok(ForeignMut {
            pointer,
            _lifetime: PhantomData,
            _not_send_sync: PhantomData,
        })
    }
}

fn validate_foreign<T: Sized>(
    memory: &(impl MemoryMap + ?Sized),
    address: Address,
    required: Access,
) -> Result<(), ForeignReferenceError> {
    if address.is_null() {
        return Err(ForeignReferenceError::NullAddress);
    }
    let len = size_of::<T>();
    if len == 0 {
        return Err(ForeignReferenceError::ZeroSizedType);
    }
    if len > isize::MAX as usize {
        return Err(ForeignReferenceError::SizeExceeded {
            size: len,
            limit: isize::MAX as usize,
        });
    }
    let alignment = align_of::<T>();
    address
        .require_alignment(alignment)
        .map_err(|source| ForeignReferenceError::Memory { source })?;
    let range = address
        .checked_range(len)
        .map_err(|source| ForeignReferenceError::Memory { source })?;
    memory
        .check_range(range, required, DEFAULT_MAX_REGIONS)
        .map_err(|source| ForeignReferenceError::Memory { source })
}

/// Checkable failures while acquiring a foreign-reference capability.
#[derive(Debug, Error, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum ForeignReferenceError {
    #[error("a foreign reference cannot be constructed from a null address")]
    NullAddress,
    #[error("a foreign reference to a zero-sized type has no range to validate")]
    ZeroSizedType,
    #[error("foreign pointee size {size} exceeds reference limit {limit}")]
    SizeExceeded { size: usize, limit: usize },
    #[error("foreign reference validation failed")]
    Memory {
        #[source]
        source: MemoryError,
    },
}

/// A scoped shared reference to storage whose validity was established by an
/// unsafe constructor.
pub struct ForeignRef<'a, T: ?Sized> {
    pointer: NonNull<T>,
    _lifetime: PhantomData<&'a T>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl<T: ?Sized> Copy for ForeignRef<'_, T> {}

impl<T: ?Sized> Clone for ForeignRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: ?Sized> ForeignRef<'a, T> {
    pub fn as_ptr(self) -> *const T {
        self.pointer.as_ptr()
    }
}

impl<T: Copy> ForeignRef<'_, T> {
    pub fn read_copy(self) -> T {
        *self
    }
}

impl<'a, T: ?Sized> From<&'a T> for ForeignRef<'a, T> {
    fn from(value: &'a T) -> Self {
        Self {
            pointer: NonNull::from(value),
            _lifetime: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

impl<T: ?Sized> Deref for ForeignRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: every constructor ties the pointer to a valid shared borrow.
        unsafe { self.pointer.as_ref() }
    }
}

/// A scoped exclusive reference to storage whose validity was established by
/// an unsafe constructor.
pub struct ForeignMut<'a, T: ?Sized> {
    pointer: NonNull<T>,
    _lifetime: PhantomData<&'a mut T>,
    _not_send_sync: PhantomData<Rc<()>>,
}

impl<'a, T: ?Sized> ForeignMut<'a, T> {
    pub fn as_ptr(&self) -> *mut T {
        self.pointer.as_ptr()
    }
}

impl<'a, T: ?Sized> From<&'a mut T> for ForeignMut<'a, T> {
    fn from(value: &'a mut T) -> Self {
        Self {
            pointer: NonNull::from(value),
            _lifetime: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

impl<T: ?Sized> Deref for ForeignMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: every constructor ties the pointer to a valid unique borrow.
        unsafe { self.pointer.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for ForeignMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: every constructor ties the pointer to a valid unique borrow.
        unsafe { self.pointer.as_mut() }
    }
}

#[cfg(any(test, feature = "test-support"))]
pub mod testing;

pub mod read;
pub use read::{DirectMemoryReadExt, FaultContainedMemoryReadExt, ReadError, ReadLimits};

#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

#[cfg(all(target_os = "macos", feature = "fault-contained"))]
pub use macos::FaultContainedProcessMemory;
#[cfg(all(target_os = "windows", feature = "fault-contained"))]
pub use windows::FaultContainedProcessMemory;

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn checked_address_arithmetic_rejects_overflow() {
        assert_eq!(
            Address::new(usize::MAX).checked_add(1),
            Err(MemoryError::AddressOverflow)
        );
        assert_eq!(
            Address::NULL.checked_sub(1),
            Err(MemoryError::AddressOverflow)
        );
        assert_eq!(
            Address::NULL.checked_add_signed(-1),
            Err(MemoryError::AddressOverflow)
        );
    }

    #[test]
    fn ranges_are_non_empty_and_half_open() {
        assert_eq!(
            AddressRange::new(Address::new(4), 0),
            Err(MemoryError::ZeroLength)
        );
        let range = AddressRange::new(Address::new(4), 4).unwrap();
        assert!(range.contains(Address::new(4)));
        assert!(range.contains(Address::new(7)));
        assert!(!range.contains(Address::new(8)));
    }

    #[test]
    fn alignment_must_be_a_nonzero_power_of_two() {
        assert_eq!(
            Address::new(8).is_aligned_to(0),
            Err(MemoryError::InvalidAlignment { alignment: 0 })
        );
        assert_eq!(
            Address::new(8).is_aligned_to(3),
            Err(MemoryError::InvalidAlignment { alignment: 3 })
        );
        assert!(Address::new(8).is_aligned_to(8).unwrap());
        assert!(!Address::new(9).is_aligned_to(8).unwrap());
    }

    struct Regions(Vec<MemoryRegion>);

    impl MemoryMap for Regions {
        fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
            self.0
                .iter()
                .copied()
                .find(|region| region.contains(address))
                .ok_or(MemoryError::Unmapped { address })
        }
    }

    fn region(start: usize, len: usize, access: Access) -> MemoryRegion {
        MemoryRegion {
            range: AddressRange::new(Address::new(start), len).unwrap(),
            access,
            kind: RegionKind::Private,
            module: None,
        }
    }

    #[test]
    fn complete_range_walks_adjacent_regions() {
        let map = Regions(vec![
            region(0x1000, 0x10, Access::READ),
            region(0x1010, 0x10, Access::READ | Access::WRITE),
        ]);
        let requested = AddressRange::new(Address::new(0x1008), 0x18).unwrap();
        assert_eq!(map.check_range(requested, Access::READ, 2), Ok(()));
        assert_eq!(
            map.check_range(requested, Access::WRITE, 2),
            Err(MemoryError::PermissionDenied {
                address: Address::new(0x1008),
                required: Access::WRITE,
                actual: Access::READ,
            })
        );
    }

    #[test]
    fn complete_range_rejects_gaps_and_region_limit() {
        let map = Regions(vec![
            region(0x1000, 0x10, Access::READ),
            region(0x1020, 0x10, Access::READ),
        ]);
        let requested = AddressRange::new(Address::new(0x1008), 0x20).unwrap();
        assert_eq!(
            map.check_range(requested, Access::READ, 2),
            Err(MemoryError::Unmapped {
                address: Address::new(0x1010)
            })
        );

        let adjacent = Regions(vec![
            region(0x1000, 0x10, Access::READ),
            region(0x1010, 0x10, Access::READ),
        ]);
        assert_eq!(
            adjacent.check_range(requested, Access::READ, 1),
            Err(MemoryError::RegionLimitExceeded { limit: 1 })
        );
    }

    struct StableMemory {
        region: MemoryRegion,
    }

    impl StableMemory {
        fn around<T>(value: &T, access: Access) -> Self {
            let start = Address::new(value as *const T as usize);
            Self {
                region: MemoryRegion {
                    range: start.checked_range(size_of::<T>()).unwrap(),
                    access,
                    kind: RegionKind::Private,
                    module: None,
                },
            }
        }

        fn bytes(bytes: &[u8], access: Access) -> Self {
            Self {
                region: MemoryRegion {
                    range: Address::new(bytes.as_ptr() as usize)
                        .checked_range(bytes.len())
                        .unwrap(),
                    access,
                    kind: RegionKind::Private,
                    module: None,
                },
            }
        }
    }

    impl MemoryMap for StableMemory {
        fn region_containing(&self, address: Address) -> Result<MemoryRegion, MemoryError> {
            if self.region.contains(address) {
                Ok(self.region)
            } else {
                Err(MemoryError::Unmapped { address })
            }
        }
    }

    impl DirectMemoryAccess for StableMemory {
        unsafe fn read_bytes_into(
            &self,
            address: Address,
            output: &mut [u8],
        ) -> Result<(), MemoryError> {
            let range = address.checked_range(output.len())?;
            self.check_range(range, Access::READ, DEFAULT_MAX_REGIONS)?;
            // SAFETY: this test backend is used only with stable Rust-owned
            // allocations that cover the validated numeric range.
            unsafe {
                std::ptr::copy(
                    address.get() as *const u8,
                    output.as_mut_ptr(),
                    output.len(),
                )
            };
            Ok(())
        }

        unsafe fn write_bytes_from(
            &self,
            address: Address,
            bytes: &[u8],
        ) -> Result<(), MemoryError> {
            let range = address.checked_range(bytes.len())?;
            self.check_range(range, Access::WRITE, DEFAULT_MAX_REGIONS)?;
            // SAFETY: same controlled-allocation contract as the read method.
            unsafe { std::ptr::copy(bytes.as_ptr(), address.get() as *mut u8, bytes.len()) };
            Ok(())
        }
    }

    #[test]
    fn scoped_copy_reads_owned_values() {
        let value = 0x1122_3344_u32;
        let memory = StableMemory::around(&value, Access::READ);
        let address = Address::new(&value as *const u32 as usize);
        let copied = unsafe {
            with_direct_memory_scope(&memory, |scope| scope.read_value::<u32>(address).unwrap())
        };
        assert_eq!(copied, value);
    }

    #[test]
    fn scoped_copy_rejects_output_overlap() {
        let mut bytes = [1_u8, 2, 3, 4];
        let memory = StableMemory::bytes(&bytes, Access::READ);
        let address = Address::new(bytes.as_ptr() as usize);
        let result = unsafe {
            with_direct_memory_scope(&memory, |scope| scope.read_bytes_into(address, &mut bytes))
        };
        assert!(matches!(result, Err(MemoryError::OverlappingRanges { .. })));
    }

    #[test]
    fn foreign_ref_dereferences_stable_owned_storage() {
        let value = 42_u64;
        let memory = StableMemory::around(&value, Access::READ);
        let address = Address::new(&value as *const u64 as usize);
        unsafe {
            with_direct_memory_scope(&memory, |scope| {
                let foreign = scope.foreign_ref::<u64>(address).unwrap();
                assert_eq!(*foreign, 42);
                assert_eq!(foreign.read_copy(), 42);
                let copied = foreign;
                assert_eq!(*copied, 42);
            })
        }
    }

    #[test]
    fn foreign_mut_updates_stable_owned_storage() {
        let mut value = 7_u32;
        let address = Address::new(&mut value as *mut u32 as usize);
        let memory = StableMemory::around(&value, Access::READ | Access::WRITE);
        unsafe {
            with_direct_memory_scope(&memory, |scope| {
                let mut foreign = scope.foreign_mut::<u32>(address).unwrap();
                *foreign = 11;
            })
        }
        assert_eq!(value, 11);
    }

    #[test]
    fn foreign_mut_requires_write_access() {
        let mut value = 7_u32;
        let address = Address::new(&mut value as *mut u32 as usize);
        let memory = StableMemory::around(&value, Access::READ);
        let rejected = unsafe {
            with_direct_memory_scope(&memory, |scope| {
                matches!(
                    scope.foreign_mut::<u32>(address),
                    Err(ForeignReferenceError::Memory {
                        source: MemoryError::PermissionDenied { .. }
                    })
                )
            })
        };
        assert!(rejected);
    }

    #[test]
    fn foreign_reference_rejects_null_misalignment_and_zst() {
        let value = 7_u32;
        let memory = StableMemory::around(&value, Access::READ);
        unsafe {
            with_direct_memory_scope(&memory, |scope| {
                assert!(matches!(
                    scope.foreign_ref::<u32>(Address::NULL),
                    Err(ForeignReferenceError::NullAddress)
                ));
                let misaligned = memory.region.range.start().checked_add(1).unwrap();
                assert!(matches!(
                    scope.foreign_ref::<u32>(misaligned),
                    Err(ForeignReferenceError::Memory {
                        source: MemoryError::Misaligned { .. }
                    })
                ));
                assert!(matches!(
                    scope.foreign_ref::<()>(memory.region.range.start()),
                    Err(ForeignReferenceError::ZeroSizedType)
                ));
            })
        }
    }

    #[test]
    fn safe_reference_conversions_follow_rust_borrows() {
        let value = 4_u16;
        assert_eq!(*ForeignRef::from(&value), 4);

        let mut mutable = 5_u16;
        {
            let mut foreign = ForeignMut::from(&mut mutable);
            *foreign = 6;
        }
        assert_eq!(mutable, 6);
    }

    proptest! {
        #[test]
        fn checked_add_matches_wide_integer_model(base: usize, offset: usize) {
            let expected = (base as u128)
                .checked_add(offset as u128)
                .filter(|value| *value <= usize::MAX as u128)
                .map(|value| value as usize);
            let actual = Address::new(base).checked_add(offset).ok().map(Address::get);
            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn checked_sub_matches_wide_integer_model(base: usize, offset: usize) {
            let expected = (base as u128)
                .checked_sub(offset as u128)
                .map(|value| value as usize);
            let actual = Address::new(base).checked_sub(offset).ok().map(Address::get);
            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn checked_signed_add_matches_wide_integer_model(base: usize, offset: i64) {
            let expected = (base as i128)
                .checked_add(offset as i128)
                .filter(|value| *value >= 0 && *value <= usize::MAX as i128)
                .map(|value| value as usize);
            let actual = Address::new(base)
                .checked_add_signed(offset)
                .ok()
                .map(Address::get);
            prop_assert_eq!(actual, expected);
        }

        #[test]
        fn ranges_match_half_open_wide_model(start: usize, len in 0usize..4096, probe: usize) {
            let expected_end = (start as u128)
                .checked_add(len as u128)
                .filter(|end| len != 0 && *end <= usize::MAX as u128);
            let actual = AddressRange::new(Address::new(start), len);
            match expected_end {
                Some(end) => {
                    let range = actual.expect("wide model predicted a representable range");
                    prop_assert_eq!(range.end().get(), end as usize);
                    let expected_contains = start <= probe && (probe as u128) < end;
                    prop_assert_eq!(range.contains(Address::new(probe)), expected_contains);
                }
                None => prop_assert!(actual.is_err()),
            }
        }
    }
}
