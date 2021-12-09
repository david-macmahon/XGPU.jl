"""
Interface to XGPU library.  See:

  - Info
  - Context
  - xgpuVersionString()
  - xgpuInit()
  - xgpuFree()
  - xgpuClearDeviceIntegrationBuffer()
  - xgpuDumpDeviceIntegrationBuffer()
  - xgpuCudaXengine()
  - xgpuReorderMatrix()
  - xgpuSwizzleInput!()
  - xgpuSwizzleRawInput!()

Also includes some convenience and general purpose reordering functions:

  - xgpuInputDims()
  - xgpuInputArray()
  - xgpuOutputDims()
  - xgpuOutputArray()
  - xgpuInputPairIndex()
  - rawSwizzleInput!()

The name of XGPU library that will be used can be specified by
`ENV["LIBXGPU"]` and defaults to `libxgpu` if not specified.  If
`ENV["LIBXGPU"]` is not an absolute path name or if the `libxgpu` default is
used, then the paths in the array `DL_LOAD_PATH` are searched first, followed
by the system load path.
"""
module XGPU

using Libdl

export xgpuinfo
export Context
export xgpuVersionString
export xgpuInit
export xgpuFree
export xgpuClearDeviceIntegrationBuffer
export xgpuDumpDeviceIntegrationBuffer
export xgpuCudaXengine
export xgpuReorderMatrix
export xgpuSwizzleInput!
export xgpuSwizzleRawInput!

export xgpuInputDims
export xgpuInputArray
export xgpuOutputDims
export xgpuOutputArray
export xgpuInputPairIndex
export rawSwizzleInput!

# Used to indicate the size and type of input data
const INT8    = (0)
const FLOAT32 = (1)
const INT32   = (2)

# Used to indicate matrix ordering
const TRIANGULAR_ORDER               = 1000
const REAL_IMAG_TRIANGULAR_ORDER     = 2000
const REGISTER_TILE_TRIANGULAR_ORDER = 3000

# Flags for xgpuInit
const DEVICE_MASK          = ((1<<16)-1)
const DONT_REGISTER_ARRAY  = (1<<16)
const DONT_REGISTER_MATRIX = (1<<17)
const DONT_REGISTER        = (DONT_REGISTER_ARRAY |
                              DONT_REGISTER_MATRIX)

# Create empty Dict that will map Symbols to callable function pointers.
const LIBXGPU = Dict{Symbol, Ptr{Cvoid}}()

# List of Tuple{Symbol, Bool} for functions that we want to wrap.  The symbol
# gives the function name and the boolean flag indicates whethere an error
# should be thrown if the symbol is not found (e.g. for optional/experimental
# functions in libxgpu).
const LIBXGPU_SYMS = Tuple{Symbol, Bool}[
                       (:xgpuInfo,                         true),
                       (:xgpuVersionString,                true),
                       (:xgpuInit,                         true),
                       (:xgpuFree,                         true),
                       (:xgpuClearDeviceIntegrationBuffer, true),
                       (:xgpuDumpDeviceIntegrationBuffer,  true),
                       (:xgpuCudaXengine,                  true),
                       (:xgpuReorderMatrix,                true),
                       (:xgpuSwizzleInput,                 true),
                       (:xgpuSwizzleRawInput,             false)
                     ]

"""
Info is a mutable struct used to convey the compile-time X engine sizing
parameters of the XGPU library.  Corresponds to an `XGPUInfo` struct.
"""
mutable struct Info
  """
  Number of polarizations (NB: will be rolled into a new "ninputs" field)
  """
  npol::UInt32
  """
  # Number of stations (NB: will be rolled into a new "ninputs" field)
  """
  nstation::UInt32
  """
  Number of baselines (derived from nstation)
  """
  nbaseline::UInt32
  """
  Number of frequencies
  """
  nfrequency::UInt32
  """
  Number of per-channel time samples per integration
  """
  ntime::UInt32
  """
  Number of per-channel time samples per transfer to GPU
  """
  ntimepipe::UInt32
  """
  Integer value indicating type of input.  One of INT8, FLOAT32, INT32.
  """
  input_type::UInt32
  """
  Integer value indicating type of computation.  One of INT8 or FLOAT32.
  """
  compute_type::UInt32
  """
  Number of ComplexInput elements in input array.  xGPU refers to the input
  array as a vector.
  """
  vecLength::UInt64
  """
  Number of ComplexInput elements per transfer to GPU.
  """
  vecLengthPipe::UInt64
  """
  Number of Complex elements in output array.  xGPU refers to the output array
  as a matrix.
  """
  matLength::UInt64
  """
  Number of Complex elements in "triangular order" output vector.
  """
  triLength::UInt64
  """
  Integer value representing the output matrix order.  One of TRIANGULAR_ORDER,
  REAL_IMAG_TRIANGULAR_ORDER, REGISTER_TILE_TRIANGULAR_ORDER.
  """
  matrix_order::UInt32
  """
  Size of each shared memory transfer
  """
  shared_atomic_size::UInt
  """
  Number of complex values per real/imag block
  """
  complex_block_size::UInt

  """
  Construct uninitialized `Info` instance (to be populated from xGPU library).
  """
  function Info()
    info = new()
    # Only initialize object from library if it's loaded
    if haskey(LIBXGPU, :xgpuInfo)
      @ccall $(LIBXGPU[:xgpuInfo])(info::Ref{Info})::Cvoid
    end
    info
  end
end

const xgpuinfo = Info()

# Module initization function to load libxgpu at runtime
function __init__()
	libfile = get(ENV, "LIBXGPU", "libxgpu")
  handle = dlopen(libfile)
  # Look up symbols
  for (sym, throw_error) in LIBXGPU_SYMS
    LIBXGPU[sym] = dlsym(handle, sym; throw_error)
  end
  # Populate xgpuinfo
  @ccall $(LIBXGPU[:xgpuInfo])(xgpuinfo::Ref{Info})::Cvoid
end

"""
Union of possible input types
"""
ComplexInput = Union{Complex{Int8}, Complex{Int32}, Complex{Float32}}

"""
Union of possible output types
"""
ComplexOutput = Union{Complex{Int32}, Complex{Float32}}

"""
mutable struct corresponding to an `XGPUContext` struct.
"""
mutable struct Context{Tin,Tout}
  """
  Pointer to input memory buffer on host
  """
  array_h::Ptr{Tin}

  """
  Pointer to output memory buffer on host
  """
  matrix_h::Ptr{Tout}

  """
  Size of input memory buffer on host in elements of `Tin`
  """
  array_len::Csize_t

  """
  Size of output memory buffer on host in elements of `Tout`
  """
  matrix_len::Csize_t

  """
  Offsets into memory buffers on host.  When calling `xgpuSetHostInputBuffer()`
  or `xgpuSetHostOutputBuffer()` (or functions that call them such as
  `xgpuInit()`), these fields are initialized to 0.  When using oversized
  externally (i.e.  caller) allocated host buffers, these fields should be set
  appropriately prior to calling `xgpuCudaXengine()`.  This is units of
  elements.  The offset in units of bytes is `input_offset * sizeof(Tin)`.
  """
  input_offset::Csize_t

  """
  Offsets into memory buffers on host.  When calling `xgpuSetHostInputBuffer()`
  or `xgpuSetHostOutputBuffer()` (or functions that call them such as
  `xgpuInit()`), these fields are initialized to 0.  When using oversized
  externally (i.e.  caller) allocated host buffers, these fields should be set
  appropriately prior to calling `xgpuCudaXengine()`.  This is units of
  elements.  The offset in units of bytes is `output_offset * sizeof(Tout)`.
  """
  output_offset::Csize_t

  # For xGPU library's internal use only
  internal::Ptr{Cvoid}

  """
  Parameterized inner constructor for xGPU-allocated host memory
  """
  function Context{Tin, Tout}()::Context{Tin, Tout} where {Tin<:ComplexInput, Tout<:ComplexOutput}
    new{Tin, Tout}(C_NULL, C_NULL, 0, 0, 0, 0, C_NULL)
  end
end

"""
    Context(info::Info=xgpuinfo)

Construct a `Context` instance based on `info`.
"""
function Context(info::Info=xgpuinfo)::Context
  if info.input_type == INT8
    Ti = Complex{Int8}
  elseif info.input_type == FLOAT32
    Ti = Complex{Float32}
  elseif info.input_type == INT32
    Ti = Complex{Int32}
  else
    error("Unknown input type: $(info.input_type)")
  end
  if info.compute_type == INT8
    To = Complex{Int32}
  elseif info.compute_type == FLOAT32
    To = Complex{Float32}
  else
    error("Unknown compute type: $(info.compute_type)")
  end
  Context{Ti, To}()
end

# Return values from xgpuCudaXengine()
const OK                          = (0)
const OUT_OF_MEMORY               = (1)
const CUDA_ERROR                  = (2)
const INSUFFICIENT_TEXTURE_MEMORY = (3)
const NOT_INITIALIZED             = (4)
const HOST_BUFFER_NOT_SET         = (5)

# Values for xgpuCudaXengine's syncOp parameter
const SYNCOP_NONE           = 0
const SYNCOP_DUMP           = 1
const SYNCOP_SYNC_TRANSFER  = 2
const SYNCOP_SYNC_COMPUTE   = 3

# Flags for xgpuInit
const DEVICE_MASK          = Int((1<<16)-1)
const DONT_REGISTER_ARRAY  = Int(1<<16)
const DONT_REGISTER_MATRIX = Int(1<<17)
const DONT_REGISTER        = (DONT_REGISTER_ARRAY |
                              DONT_REGISTER_MATRIX)

"""
    xgpuVersionString()::VersionNumber

Return VersionNumber correspinding to XGPU library version string.
"""
function xgpuVersionString()::VersionNumber
  # const char * xgpuVersionString();
  VersionNumber(unsafe_string(@ccall $(LIBXGPU[:xgpuVersionString])()::Cstring))
end

"""
    xgpuInit(context::Context=Context(), device_flags::Int=0)::Context

Initialize the xGPU library.

In addition to allocating device memory and initializing private internal
context, this routine calls `xgpuSetHostInputBuffer()` and
`xgpuSetHostOutputBuffer()`.  Be sure to set the context's `array_h` and
`matrix_h` fields and the corresponding length fields accordingly.

If `context.array_h` is `C_NULL`, an array of `ComplexInput` elements is
allocated (of the appropriate size) via `CudaMallocHost()`, otherwise the
memory region pointed to by `context.array_h` of length `context.array_len` is
registered with CUDA via the `CudaHostRegister()` function.

If `context.matrix_h` is `C_NULL`, an array of `ComplexOutput` elements is
allocated (of the appropriate size) via `CudaMallocHost`, otherwise the memory
region pointed to by `context.matrix_h` of length `context.matrix_len` if
registered with CUDA via the `CudaHostRegister()` function.

The index of the GPU device should be put into `device_flags`. In addition,
the following optional flags can be bitwise or'd into this value:

| Flag name              | Purpose                                        |
|:-----------------------|:-----------------------------------------------|
| `DONT_REGISTER_ARRAY`  | Disables registering (pinning) of host array   |
| `DONT_REGISTER_MATRIX` | Disables registering (pinning) of host matrix  |
| `DONT_REGISTER`        | Disables registering (pinning) of all host mem |

E.g.: `gpuInit(context, device_idx | DONT_REGISTER_ARRAY)`

Note that if registering is disabled, the corresponding `xgpuSetHost*Buffer()`
function _must_ be called prior to calling `xgpuCudaXengine()`.
"""
function xgpuInit(context::Context=Context(), device_flags::Int=0)::Context
  # int xgpuInit(XGPUContext *context, int device_flags);
  rc = @ccall $(LIBXGPU[:xgpuInit])(context::Ref{Context},
                                    device_flags::Cint)::Cint
  if rc != OK
    if rc == OUT_OF_MEMORY
      throw(OutOfMemoryError())
    elseif rc == INSUFFICIENT_TEXTURE_MEMORY
      error("insufficient texture memory")
    elseif rc == CUDA_ERROR
      error("CUDA error")
    elseif rc == NOT_INITIALIZED
      error("not initialized")
    elseif rc == HOST_BUFFER_NOT_SET
      error("host buffer not set")
    else
      error("unexpected error $(rc)")
    end
  end

  context
end

"""
    xgpuFree(context::Context)::Nothing

Frees library-allocated memory on host and device.
"""
function xgpuFree(context::Context)::Nothing
  #  void xgpuFree(XGPUContext *context);
  @ccall $(LIBXGPU[:xgpuFree])(context::Ref{Context})::Cvoid
end

"""
    xgpuClearDeviceIntegrationBuffer(context::Context)::Nothing

Clear the device integration buffer.

Sets the device integration buffer associated with `context` to all zeros,
effectively starting a new integration.
"""
function xgpuClearDeviceIntegrationBuffer(context::Context)::Nothing
  # int xgpuClearDeviceIntegrationBuffer(XGPUContext *context);
  rc = @ccall $(LIBXGPU[:xgpuClearDeviceIntegrationBuffer])(context::Ref{Context})::Cint
  if rc != OK
    if rc == NOT_INITIALIZED
      error("not initialized error")
    else
      error("unexpected error $(rc)")
    end
  end
  nothing
end

"""
    xgpuDumpDeviceIntegrationBuffer(context::Context;
                                    reorder::Bool=true)::Nothing

Wait for all GPU transfer/compute activity to complete and then copy the
integration buffer from device memory to host memory at `(context.matrix_h +
context.output_offset)`.  This is provided as an alternative to passing
SYNCOP_DUMP to xgpuCudaXengine().
"""
function xgpuDumpDeviceIntegrationBuffer(context::Context;
                                         reorder::Bool=true)::Nothing
  # int xgpuDumpDeviceIntegrationBuffer(XGPUContext *context);
  rc = @ccall $(LIBXGPU[:xgpuDumpDeviceIntegrationBuffer])(context::Ref{Context})::Cint
  if rc != OK
    if rc == NOT_INITIALIZED
      error("not initialized error")
    else
      error("unexpected error $(rc)")
    end
  end

  if reorder
    xgpuReorderMatrix(context)
  end

  nothing
end

"""
    xgpuCudaXengine(context::Context, syncop::Int;
                    reorder::Bool=true)::Nothing

Perform correlation.  Correlates the input data at `context.array_h +
context.input_offset`.  The `syncop` parameter specifies what will be done
after sending all the asynchronous tasks to CUDA.  The possible values and
their meanings are:

    SYNCOP_NONE - No further action is taken.
    SYNCOP_DUMP - Waits for all transfers and computations to
                  complete, then dumps to output buffer at
                  `context.matrix_h + context.output_offset`.
    SYNCOP_SYNC_TRANSFER - Waits for all transfers to complete,
                           but not necessrily all computations.
    SYNCOP_SYNC_COMPUTE  - Waits for all computations (and transfers) to
                           complete, but does not dump.

If `syncop` is `SYNCOP_DUMP`, the output buffer at `context.matrix_h +
context.output_offset` will be reordered into the station pair triangualar
order supported by `xgpuInputPairIndex()` unless the caller passes keyword
argument `reorder=false`.
"""
function xgpuCudaXengine(context::Context, syncop::Int;
                         reorder::Bool=true)::Nothing
  # int xgpuCudaXengine(XGPUContext *context, int syncOp);
  rc = @ccall $(LIBXGPU[:xgpuCudaXengine])(context::Ref{Context},
                                           syncop::Cint)::Cint
  if rc != OK
    if rc == NOT_INITIALIZED
      error("not initialized")
    elseif rc == HOST_BUFFER_NOT_SET
      error("host buffer not set")
    elseif rc == OUT_OF_MEMORY
      throw(OutOfMemoryError())
    elseif rc == INSUFFICIENT_TEXTURE_MEMORY
      error("insufficient texture memory")
    elseif rc == CUDA_ERROR
      error(" CUDA error")
    else
      error("unexpected error $(rc)")
    end
  end

  if syncop == SYNCOP_DUMP && reorder
    xgpuReorderMatrix(context)
  end

  nothing
end

"""
    xgpuReorderMatrix(context::Context)::Nothing

This function will reorder the output buffer from the convoluted so-called
"register tile triangular order" to a somewhat simpler "triangular order"
supported by `xgpuInputPairIndex()`.  When in "triangular order" the `npol^2`
cross products for pairs of stations are grouped together and these groups are
ordered in the order of the non-redundant triangular half (including the
auto-correlation diagonal) of the Hermitian correlation matrix.  When
`npol==2`, the cross product group for a station's auto-correlation will
include a redundant cross product, but the benefit from keeping the auto- and
cross-correlation groups sized the same far outweighs the cost of the
negligible extra storage.

The triangular order for the station pair groups are shown in this table:

| Sj \\ Si | 1 | 2 | 3 | ⋯ |
|:--------:|:-:|:-:|:-:|:-:|
|    1     | 1 |   |   |   |
|    2     | 2 | 3 |   |   |
|    3     | 4 | 5 | 6 |   |
|    ⋮     | ⋮ | ⋮ | ⋮ | ⋱ |

The group for station 2 paired with itself (i.e. `Si == Sj == 2`) is the third
group.  The group for station 2 paired with station 3 (i.e. `Si == 2, Sj == 3`)
is the fifth group.  Each group contains `npol^2` cross products: 1 when `npol
== 1`, 4 when `npol == 2`.  To keep the dimensioning of the output matrix the
same regardless of `npol`, the extra dimensionality when `npol==2` is
flattened.

When `npol == 2`, the cross products within the group for station `Si` and
`Sj`, with `i <= j` are ordered as:

    SjP1 * conj(SiP1)
    SjP1 * conj(SiP2)
    SjP2 * conj(SiP1)
    SjP2 * conj(SiP2)

This function will be called automatically whenever `xgpuCudaXengine() is
passed `SYNCOP_DUMP`, unless the caller also explicitly passes `reorder=false`.
"""
function xgpuReorderMatrix(context::Context{Tin,Tout})::Nothing where {Tin<:ComplexInput, Tout<:ComplexOutput}
  # Julia pointer arithmetic is always in bytes!
  matrix_ptr = context.matrix_h + context.output_offset * sizeof(Tout)
  # void xgpuReorderMatrix(Complex *matrix);
  @ccall $(LIBXGPU[:xgpuReorderMatrix])(matrix_ptr::Ptr{Tout})::Cvoid
end

"""
    xgpuSwizzleInput!(zout::Array{Complex{Int8}},
                       zin::Array{Complex{Int8}})::Nothing

Reorder input array `zin` from xGPU canonical input ordering to output array
`zout` in xGPU swizzled input ordering used when xGPU is compiled to utilize
the DP4A feature.

This is a weird function because its parameters are declared as
(ComplexInput *), but internally it only treats them as (signed char *).
For this reason, the method defined here only accepts `Array(Complex{Int8}}`.

The "standard" ordering for an xGPU input `Array{Complex{Tin}}` is `(P,S,F,T)`,
where `P` is the number of polarizations (`npol`), `S` is the number of
stations (aka antennas, `nstation`), `F` is the number of frequency channels
(`nfrequency`), and `T` is the number of time samples (`ntime`).  The swizzle
operation essentially performs the following sequence of operations, assuming
`size(zin) == size(zout) == (P, S, F, T)`:

| Operation      | Element Type  | Dimensions         |
|:---------------|:--------------|:-------------------|
| original input | Complex{Int8} | (P, S, F, T)       |
| reinterpret    | Int8          | (2P, S, F, T)      |
| reshape        | Int8          | (2P, S, F, 4, T÷4) |
| permutedims    | Int8          | (4, 2P, S, F, T÷4) |
| reinterpret    | Complex{Int8} | (2, 2P, S, F, T÷4) |
| reshape        | Complex{Int8} | (P, S, F, T)       |

Of course, the dimensions of `zin` and `zout` are not actually changed by this
call and can be anything so long as they contain (at least) the expected number
of bytes, though adhering to the prescribed dimensionality is recommended.

Even though the element type of the output Array is `Complex{Int8}`, they are
really just pairs of real componnts and pairs of imaginary components rather
than being actual complex values.
"""
function xgpuSwizzleInput!(zout::Array{Complex{Int8}},
                            zin::Array{Complex{Int8}})::Nothing
  # void xgpuSwizzleInput(ComplexInput *out, const ComplexInput *in);
  @ccall $(LIBXGPU[:xgpuSwizzleInput])(zout::Ref{Complex{Int8}},
                                       zin::Ref{Complex{Int8}})::Cvoid
end

"""
    xgpuSwizzleRawInput!(zout::Array{Complex{Int8}},
                          zin::Array{Complex{Int8}},
                          tstart=1, tstride=size(zon,2))::Nothing

Reorders `zin` in GUPPI RAW format, starting at 1-based time sample `tstart`,
to xGPU swizzled input ordered `zout`.  Separate real/imag and corner turn in
time, depth 4.

The GUPPI RAW ordering of a `Array{Complex{Int8}}` Array is `(P,T,F,S)`, where
`P` is the number of polarizations (`npol`), `T` is the number of time samples
(`ntime`), `F` is the number of frequency channels (`nfrequency`), and `S` is
the number of stations (aka antennas, `nstation`).  The swizzle operation
essentially performs the following sequence of operations, assuming `size(zin)
== (P, T, F, S)` and `size(zout) == (P, S, F, T)`:

| Operation      | Element Type  | Dimensions         |
|:---------------|:--------------|:-------------------|
| original input | Complex{Int8} | (P, T, F, S)       |
| reinterpret    | Int8          | (2P, T, F, S)      |
| reshape        | Int8          | (2P, 4, T÷4, F, S) |
| permutedims    | Int8          | (4, 2P, S, F, T÷4) |
| reinterpret    | Complex{Int8} | (2, 2P, S, F, T÷4) |
| reshape        | Complex{Int8} | (P, S, F, T)       |

Of course, the dimensions of `zin` and `zout` are not actually changed by this
call and can be anything so long as they contain (at least) the expected number
of bytes, though adhering to the prescribed dimensionality is recommended.

Even though the element type of the output Array is `Complex{Int8}`, they are
really just pairs of real componnts and pairs of imaginary components rather
than being actual complex values.

!!! warning
    This is an experimental function that does not yet exist in public xGPU!!!
"""
function xgpuSwizzleRawInput!(zout::Array{Complex{Int8}},
                              zin::Array{Complex{Int8}},
                              tstart=1,
                              tstride=size(zin,2))::Nothing
  # Make sure library has xgpuSwizzleRawInput function
  if !haskey(LIBXGPU, :xgpuSwizzleRawInput)
    @error "xgpuSwizzleRawInput not present in libxgpu"
  end

  # void xgpuSwizzleRawInput(ComplexInput *out, const ComplexInput *in, size_t tstride);
  @ccall $(LIBXGPU[:xgpuSwizzleRawInput])(zout::Ref{Complex{Int8}},
                                          Ref(zin,tstart)::Ref{Complex{Int8}},
                                          tstride::Csize_t)::Cvoid
end

### Julia code below here uses code above rather than libxgpu directly.

"""
    xgpuInputDims(info::Info=xgpuinfo)::NTuple{4,UInt32}

Convenience function to return a tuple corresponding to the dimensions of the
xGPU host input array based on information in `info`:

    (info.npol, info.nstation, info.nfrequency, info.ntime)
"""
function xgpuInputDims(info::Info=xgpuinfo)::NTuple{4,UInt32}
  (info.npol, info.nstation, info.nfrequency, info.ntime)
end

"""
    xgpuInputArray(context::Context, info::Info=xgpuinfo)::Array

Returns a Julia Array whose data reside in the memory pointed to by
`context.array_h` and whose dimensions are given by `xgpuInputDims(info)`.
"""
function xgpuInputArray(context::Context, info::Info=xgpuinfo)::Array
  @assert context.array_h != C_NULL "xgpu host input array not allocated"
  unsafe_wrap(Array, context.array_h, xgpuInputDims(info))
end

"""
    xgpuOutputDims(info::Info=xgpuinfo)::NTuple{2,UInt32}

Convenience function to return a tuple correspinding to the dimensions of the
xGPU host output array based on information in `info`:

    (nstation*(nstation+1)÷2*npol^2, info.nfrequency)

The dimensions returned assume/require that the output array has been reordered
by an explicit call to `xgpuReorderMatrix()` after a `SYNCOP_DUMP`.

Notice that the first dimension is the number of unique station pairings,
sometimes referred to as baselines, times the number of polarizations squared.
When `npol==2`, each station will have one conjugate-redundant pairing with
itself.

xGPU makes a distinction between "station" (aka "antenna") and "polarization",
the "(station, polarization)" tuple is just another way of representing a
single input `(station-1)*npol + polarization)`, where station=1:nstation and
polarization=1:npol.  See `xgpuInputPairIndex()` for more details.
"""
function xgpuOutputDims(info::Info=xgpuinfo)::NTuple{2,UInt32}
  ((info.nstation * (info.nstation+1)) ÷ 2 * info.npol^2, info.nfrequency)
end

"""
    xgpuOutputArray(context::Context, info::Info=xgpuinfo)::Array

Returns a Julia Array whose data reside in the memory pointed to by
`context.array_h` and whose dimensions are given by `xgpuInputDims(info)`.

After a `SYNCOP_DUMP`, the xGPU output array must be reordered by a call to
`xgpuReorderMatrix()` otherwise the in-memory layout of the data will not match
the dimensionality of the returned Array.  This is done automatically when
`SYNCOP_DUMP` is passed to `xgpuCudaXengine()` unless the caller explicitly
passes `reorder=false`.
"""
function xgpuOutputArray(context::Context, info::Info=xgpuinfo)::Array
  @assert context.matrix_h != C_NULL "xgpu host output array not allocated"
  unsafe_wrap(Array, context.matrix_h, xgpuOutputDims(info))
end

"""
    xgpuInputPairIndex(sp1, sp2, info::Info=xgpuinfo)::Integer

Returns the linear index into the xGPU output array corresponding to the cross
product of input pair `sp1` and `sp2`.  Inputs `sp1` and `sp2` may be given as
an `Integer` or as a `(station, polarization)` tuple.  In all cases, stations
and polarizations and the return value are indexed from 1 and parameters are
bounds checked against `xgpuinfo` (via `@assert`).

The same index is returned regardless of the order of the two requested inputs.
Attention to conjugation and input ordering is important.  xGPU calculates the
correlation as:

    (conj(z[i]) * z[j]), where i <= j

This may differ from local convention.  As is always the case, conjugation
needs to be verified in the end-to-end system to ensure proper interpretation.

xGPU treats inputs as a `(station, polarization)` tuple, which implies that
input `(station=1, polarization=1)` must come from the same station/antenna as
input `(station=1, polarization=2)`, but in fact xGPU has no such requirement
that the two values come from the same physical station/antenna.

The input tuple for a given input number can be calculated as:

    station, pol = divrem(input_number-1, xgpuinfo.npol) .+ 1

!!! note
    `station` and `polarization` are merely xGPU constructs.  They are not
    necessarily representative or indicative of which antenna or feed
    polarization is the actual source of the input data.  The relationship
    between  physical antennas/feeds and xGPU inputs is the user's
    responsibility.
"""
function xgpuInputPairIndex(sp1::NTuple{2,Integer}, sp2::NTuple{2,Integer},
                            info::Info=xgpuinfo)::Integer
  s1, p1 = sp1
  s2, p2 = sp2
  @assert s1 in 1:info.nstation "station $(s1) is not in range 1:$(info.nstation)"
  @assert p1 in 1:info.npol "polarization $(p1) is not in range 1:$(info.npol)"
  @assert s2 in 1:info.nstation "station $(s2) is not in range 1:$(info.nstation)"
  @assert p2 in 1:info.npol "polarization $(p2) is not in range 1:$(info.npol)"
  if (s1-1) * info.npol + p1 > (s2-1) * info.npol + p2
    s1, p1, s2, p2 = s2, p2, s1, p1
  end
  station_pair_index = (s2*(s2-1)) ÷ 2 + s1
  pol_pair_index = (p2-1) * info.npol + p1
  (station_pair_index-1) * info.npol^2 + pol_pair_index
end

function xgpuInputPairIndex(i1::Integer, sp2::NTuple{2,Integer},
                            info::Info=xgpuinfo)::Integer
  sp1 = divrem(i1 - 1, info.npol) .+ 1
  xgpuInputPairIndex(sp1, sp2)
end

function xgpuInputPairIndex(sp1::NTuple{2,Integer}, i2::Integer,
                            info::Info=xgpuinfo)::Integer
  sp2 = divrem(i2 - 1, info.npol) .+ 1
  xgpuInputPairIndex(sp1, sp2)
end

function xgpuInputPairIndex(i1::Integer, i2::Integer,
                            info::Info=xgpuinfo)::Integer
  sp1 = divrem(i1 - 1, info.npol) .+ 1
  sp2 = divrem(i2 - 1, info.npol) .+ 1
  xgpuInputPairIndex(sp1, sp2)
end

#= TODO Think about this some more...
"""
    getindex(matrix_h::Array, XGPU, sp1, sp2, channel)

Return `matrix_h[xgpuInputPairIndex(sp1, sp2), channel]`.

More commonly called as `matrix_h[XGPU, i1, i2, channel]`.  `i1` and `i2` can
be Integer input indexes or `(station, pol)` tuples, but not an Array.
`channel` can be Integer frequency channel number or any other index value
(e.g. Array or UnitRange or even just `:`).
"""
function Base.getindex(matrix_h::Array, XGPU, sp1, sp2, channel=:)
  matrix_h[xgpuInputPairIndex(sp1, sp2), channel]
end
=#

#= TODO This is compact/concise, but slow!
function rawSwizzleInput!(zout::Array{Complex{Int8}},
                           zin::Array{Complex{Int8}, 4})::Nothing
  # This must be true or things won't go well...
  @assert sizeof(zin) == sizeof(zout)

  # Create reinterpreted reshaped views to zin and zout
  p, t, f, s = size(zin)
  vin  = reshape(reinterpret(Int8, zin ), 2*p, 4, t÷4, f, s)
  vout = reshape(reinterpret(Int8, zout), 4, 2*p, s, f, t÷4)

  # Use permutedims! to copy data from vin to vout
  permutedims!(vout, vin, (2, 1, 5, 4, 3))

  nothing
end
=#

"""
    rawSwizzleInput!(swizout::Array{Complex{Int8}},
                     rawin::Array{Complex{Int8}, 4},
                     toffset=1,
                     info::Info=xgpuinfo,
                     rawnants::UInt32=info.nstation,
                     rawnchan::UInt32=info.nfrequency)::Nothing

Swizzle-copy data from GUPPI RAW block `rawin` to xGPU input Array `swizout`.
The `rawnants` parameter defines the number of stations in `rawin`.  The
`rawnchan` parameter defines the number of frequencies in `rawin`.  These may
be less than the number of stations or frequency channels that XGPU was
compiled for.  This allows for fewer channels and/or fewer stations to be
correlated than XGPU was compiled for, but XGPU will always perform the full
correlation of the entire input buffer (even if only partially populated).

The GUPPI RAW block may contain more that `info.ntime` time samples.  `toffset`
specifies the starting time sample of the GUPPI raw block from which to
swizzle-copy into `swizout`.  After performing the swizzle copy, the user will
typically call `xgpuCudaXengine()` to initiate transfer and correlation of the
swizzle-copied data.
"""
function rawSwizzleInput!(swizout::Array{Complex{Int8}},
                          rawin::Array{Complex{Int8}, 4},
                          toffset::Int=1,
                          info::Info=xgpuinfo,
                          rawnants::Integer=info.nstation,
                          rawnchan::Integer=info.nfrequency)::Nothing
  vrawin = reinterpret(Int8, rawin)
  vswizout = reinterpret(Int8, swizout)

  tstride = size(rawin, 2)
  @assert(toffset in 1:(tstride-info.ntime+1),
          "toffset $(toffset) not in 1:$(tstride-info.ntime+1)")

  swizout1 = unsafe_wrap(Array, pointer(vswizout), sizeof(swizout))
  rawin1 = unsafe_wrap(Array, pointer(vrawin, 4*(toffset-1)+1),
                       sizeof(rawin)-4*(toffset-1))

  Threads.@threads for f=0:rawnchan-1
    for s=0:rawnants-1
      for t=0:info.ntime-1
        offset_in = ((s*rawnchan+f)*tstride+t)*info.npol
        offset_out = ((t÷4*info.nfrequency+f)*info.nstation+s)*info.npol
        for p=0:info.npol-1
          for c=0:1
            @inbounds swizout1[((offset_out+p)*2+c)*4+t%4 + 1] =
              rawin1[(offset_in+p)*2 + c + 1];
          end
        end
      end
    end
  end
end

end # module
