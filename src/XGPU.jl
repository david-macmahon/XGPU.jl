"""
Interface to XGPU library.  See:
  - Info
  - Context
  - xgpuVersionString()
  - xgpuInit()
  - xgpuFree()
  - xgpuClearDeviceIntegrationBuffer()
  - xgpuCudaXengine()
  - xgpuSwizzleInput!()

"""
module XGPU

export LIBXGPU
export Info
export Context
export xgpuVersionString
export xgpuInit
export xgpuFree
export xgpuClearDeviceIntegrationBuffer
export xgpuCudaXengine
export xgpuSwizzleInput!

const LIBXGPU = "/home/davidm/local/src/xgpu/src/libxgpu.so"

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
  Number of ComplexInput elements in input vector
  """
  vecLength::UInt64
  """
  Number of ComplexInput elements per transfer to GPU.
  """
  vecLengthPipe::UInt64
  """
  Number of Complex elements in output vector.
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
  Construct `Info` instance and populate it from xGPU library.
  """
  function Info()
    info = new()
    @ccall LIBXGPU.xgpuInfo(info::Ref{Info})::Cvoid
    info
  end
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
  appropriately prior to calling `xgpuCudaXengine()`.
  """
  input_offset::Csize_t

  """
  Offsets into memory buffers on host.  When calling `xgpuSetHostInputBuffer()`
  or `xgpuSetHostOutputBuffer()` (or functions that call them such as
  `xgpuInit()`), these fields are initialized to 0.  When using oversized
  externally (i.e.  caller) allocated host buffers, these fields should be set
  appropriately prior to calling `xgpuCudaXengine()`.
  """
  output_offset::Csize_t

  # For xGPU library's internal use only
  internal::Ptr{Cvoid}

  """
  Parameterized inner constructor for xGPU-allocated host memory
  """
  function Context{Tin, Tout}() where {Tin<:ComplexInput, Tout<:ComplexOutput}
    new{Tin, Tout}(C_NULL, C_NULL, 0, 0, 0, 0, C_NULL)
  end
end

"""
    Context(info::Info=Info())

Construct a `Context` instance based on `info`.
"""
function Context(info::Info=Info())
  if info.input_type == INT8
    Tin = Complex{Int8}
  elseif info.input_type == FLOAT32
    Tin = Complex{Float32}
  elseif info.input_type == INT32
    Tin = Complex{Int32}
  else
    error("Unknown input type: $(info.input_type)")
  end
  if info.compute_type == INT8
    Tout = Complex{Int32}
  elseif info.compute_type == FLOAT32
    Tout = Complex{Float32}
  else
    error("Unknown compute type: $(info.compute_type)")
  end
  Context{Tin, Tout}()
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

Get pointer to library version string.

The library version string should not be modified or freed!
"""
function xgpuVersionString()::VersionNumber
  # const char * xgpuVersionString();
  VersionNumber(unsafe_string(@ccall LIBXGPU.xgpuVersionString()::Cstring))
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
  rc = @ccall LIBXGPU.xgpuInit(context::Ref{Context},
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
  @ccall LIBXGPU.xgpuFree(context::Ref{Context})::Cvoid
end

"""
    xgpuClearDeviceIntegrationBuffer(context::Context)::Nothing

Clear the device integration buffer.

Sets the device integration buffer to all zeros, effectively starting a new
integration.
"""
function xgpuClearDeviceIntegrationBuffer(context::Context)::Nothing
  # int xgpuClearDeviceIntegrationBuffer(XGPUContext *context);
  rc = @ccall LIBXGPU.xgpuClearDeviceIntegrationBuffer(context::Ref{Context})::Cint
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
    xgpuCudaXengine(context::Context, syncop::Int)::Nothing

Perform correlation.  Correlates the input data at `context->array_h +
context->input_offset`.  The `syncOp` parameter specifies what will be done
after sending all the asynchronous tasks to CUDA.  The possible values and
their meanings are:

    SYNCOP_NONE - No further action is taken.
    SYNCOP_DUMP - Waits for all transfers and computations to
                  complete, then dumps to output buffer at
                  `context->matrix_h + context->output_offset`.
    SYNCOP_SYNC_TRANSFER - Waits for all transfers to complete,
                           but not necessrily all computations.
    SYNCOP_SYNC_COMPUTE  - Waits for all computations (and transfers) to
                           complete, but does not dump.
"""
function xgpuCudaXengine(context::Context, syncop::Int)::Nothing
  # int xgpuCudaXengine(XGPUContext *context, int syncOp);
  rc = @ccall LIBXGPU.xgpuCudaXengine(context::Ref{Context},
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

  nothing
end

"""
    xgpuSwizzleInput!(zout::Array{Complex{Int8}},
                      zin::Array{Complex{Int8}})::Nothing

Reorder the input array.  Separate real/imag and corner turn in time, depth 4.

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
| reshape        | Int8          | (2P, S, F, 4, T/4) |
| permutedims    | Int8          | (4, 2P, S, F, T/4) |
| reinterpret    | Complex{Int8} | (2, 2P, S, F, T/4) |
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
  @ccall LIBXGPU.xgpuSwizzleInput(zout::Ref{Complex{Int8}},
                                   zin::Ref{Complex{Int8}})::Cvoid
end

end # module
