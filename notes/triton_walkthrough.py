"""
triton_walkthrough.py
=====================
A runnable program that shows, step by step, how Triton turns a Python function
into a GPU kernel WITHOUT the Python interpreter executing the function body.

Requirements: an NVIDIA GPU (AMD works too), `pip install torch triton`.
Run:          python triton_walkthrough.py

Read the comments top to bottom; the printed output follows the same steps.
"""

import ast
import inspect
import textwrap
import time

import torch
import triton
import triton.language as tl


def banner(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

# ---------------------------------------------------------------------------
# STEP 1: DEFINE THE KERNEL
# ---------------------------------------------------------------------------
# @triton.jit does NOT run or compile this function. Python builds the function
# object, hands it to Triton, and Triton wraps it in a `JITFunction` that just
# stores the source code and waits. The body below is never executed by CPython.
#
# The kernel is written from the point of view of ONE "program instance"
# (one block of work). The GPU will start many copies, each with a different
# program_id, and each copy handles BLOCK_SIZE elements at once.
@triton.jit
def add_kernel(
    x_ptr,                    # raw GPU address of input x
    y_ptr,                    # raw GPU address of input y
    output_ptr,               # raw GPU address of the output
    n_elements,               # runtime value: how many elements in total
    BLOCK_SIZE: tl.constexpr, # compile-time constant: elements per block
):
    # Which block am I? (0 for the first copy, 1 for the second, ...)
    pid = tl.program_id(axis=0)

    # First element this block is responsible for.
    block_start = pid * BLOCK_SIZE

    # `offsets` is a whole VECTOR of BLOCK_SIZE indices, not a single number:
    # [block_start, block_start+1, ..., block_start+BLOCK_SIZE-1].
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # The last block may run past the end of the array, so build a boolean
    # vector that marks which positions are valid.
    mask = offsets < n_elements

    # Load a whole block from GPU memory in one operation. Here x_ptr + offsets
    # is pointer arithmetic on a vector of addresses.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # One line, BLOCK_SIZE additions. The compiler spreads them over the
    # block's threads.
    output = x + y

    # Write the block back to GPU memory (only the valid positions).
    tl.store(output_ptr + offsets, output, mask=mask)


# ---------------------------------------------------------------------------
# The Python-side helper that launches the kernel.
# This function IS normal Python and IS run by the interpreter.
# ---------------------------------------------------------------------------
def add(x, y, block_size=1024):
    output = torch.empty_like(x)
    n = output.numel()

    # The grid says how many program instances to start.
    # `meta` is a dict holding the kernel's constexpr values.
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)

    # `add_kernel[grid]` calls JITFunction.__getitem__ (returns a launcher),
    # and calling that launcher with arguments calls JITFunction.run(...).
    # run() builds a cache key, compiles on a cache miss, then launches.
    # The call returns a CompiledKernel object in recent Triton versions.
    compiled = add_kernel[grid](x, y, output, n, BLOCK_SIZE=block_size)
    return output, compiled


def timed(label, fn):
    """Run fn(), wait for the GPU to finish, and print how long it took."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) * 1000
    print(f"{label:<52s} {ms:9.2f} ms")
    return result


def main():
    # -----------------------------------------------------------------------
    banner("STEP 2: What did @triton.jit actually create?")
    # -----------------------------------------------------------------------
    print("type(add_kernel) =", type(add_kernel).__name__)
    print("-> not a plain function; it is Triton's JITFunction wrapper.")
    print("   Nothing has been compiled yet. It is only holding the source.")

    # -----------------------------------------------------------------------
    banner("STEP 3: The kernel cannot be called like a normal function")
    # -----------------------------------------------------------------------
    try:
        add_kernel(None, None, None, 0, BLOCK_SIZE=16)
    except Exception as e:
        print("Calling add_kernel(...) directly raised:")
        print(f"  {type(e).__name__}: {e}")
        print("-> Python is not meant to execute the body. It must be launched")
        print("   with add_kernel[grid](...), which goes through the compiler.")

    # -----------------------------------------------------------------------
    banner("STEP 4: What Triton reads: source text -> Python AST")
    # -----------------------------------------------------------------------
    # add_kernel.fn is the original Python function that Triton stored.
    source = textwrap.dedent(inspect.getsource(add_kernel.fn))
    tree = ast.parse(source)
    print("Source text starts with:")
    print("  " + "\n  ".join(source.splitlines()[:3]) + "\n  ...")

    # Find the statement `output = x + y` in the tree and dump it. The AST
    # describes the operation; it does not compute anything.
    func_def = tree.body[0]
    for stmt in func_def.body:
        if isinstance(stmt, ast.Assign) and getattr(stmt.targets[0], "id", "") == "output":
            print("\nThe line `output = x + y` is stored as this tree:")
            print(ast.dump(stmt, indent=2))
    print("\n-> Triton's CodeGenerator walks nodes like these and, for each one,")
    print("   emits an IR operation (for BinOp(Add) on tensors: an elementwise add).")
    print("   Calls like tl.load are recognized by name and become tt.load ops.")

    # -----------------------------------------------------------------------
    # Everything below needs a GPU.
    # -----------------------------------------------------------------------
    if not torch.cuda.is_available():
        print("\nNo CUDA GPU found, so the compile-and-launch steps are skipped.")
        print("Run this on a machine with an NVIDIA (or AMD ROCm) GPU to see them.")
        return

    size = 98_432
    x = torch.rand(size, device="cuda")
    y = torch.rand(size, device="cuda")

    # -----------------------------------------------------------------------
    banner("STEP 5: First launch = compile + cache + run")
    # -----------------------------------------------------------------------
    # The first call is slow because Triton does the whole pipeline:
    #   AST -> Triton IR -> Triton GPU IR -> LLVM IR -> PTX -> cubin (SASS)
    # The binary is then stored on disk (default: ~/.triton/cache).
    out, compiled = timed("1st call (parse + compile + cache + launch)", lambda: add(x, y))
    print("Result correct:", torch.allclose(out, x + y))

    # -----------------------------------------------------------------------
    banner("STEP 6: Look at what the compiler produced")
    # -----------------------------------------------------------------------
    asm = getattr(compiled, "asm", None)
    if asm:
        print("Compilation stages available:", list(asm.keys()))
        print("\n--- Triton IR (first lines): hardware-independent, block level ---")
        print("\n".join(str(asm["ttir"]).splitlines()[:14]))
        if "ptx" in asm:
            print("\n--- PTX (first lines): NVIDIA virtual assembly ---")
            print("\n".join(str(asm["ptx"]).splitlines()[:14]))
        print("\nRegisters per thread:", getattr(compiled, "n_regs", "n/a"))
    else:
        print("This Triton version did not return a CompiledKernel from the launch.")

    # -----------------------------------------------------------------------
    banner("STEP 7: Later launches hit the cache")
    # -----------------------------------------------------------------------
    timed("2nd call, same inputs (cache hit)", lambda: add(x, y))
    timed("3rd call, same inputs (cache hit)", lambda: add(x, y))

    # A different *runtime value* (vector length) does not change the cache key
    # in general, so there is no recompile. (A divisibility hint, such as
    # n % 16 == 0, can change, which may cause one extra compile.)
    x_small, y_small = x[:5000].contiguous(), y[:5000].contiguous()
    timed("Different length (usually still a cache hit)", lambda: add(x_small, y_small))

    # -----------------------------------------------------------------------
    banner("STEP 8: Things that DO trigger a recompile")
    # -----------------------------------------------------------------------
    # A new dtype changes the instructions needed, so it needs a new binary.
    xh, yh = x.half(), y.half()
    timed("float16 inputs (new dtype -> recompile)", lambda: add(xh, yh))
    timed("float16 again (cache hit)", lambda: add(xh, yh))

    # A new constexpr value is baked into the binary, so it needs a new one.
    timed("BLOCK_SIZE=512 (new constexpr -> recompile)", lambda: add(x, y, block_size=512))
    timed("BLOCK_SIZE=512 again (cache hit)", lambda: add(x, y, block_size=512))

    # -----------------------------------------------------------------------
    banner("STEP 9: How sequential-looking code becomes parallel")
    # -----------------------------------------------------------------------
    block = 1024
    num_programs = triton.cdiv(size, block)
    print(f"Vector length      : {size}")
    print(f"BLOCK_SIZE         : {block}")
    print(f"Program instances  : {num_programs}  (the grid)")
    print(f"Last block covers  : {size - (num_programs - 1) * block} valid elements"
          f" (the mask hides the other {num_programs * block - size})")
    print("\nEach instance runs the same compiled kernel with a different")
    print("tl.program_id. Inside an instance, each line (like x + y) works on")
    print("a whole block at once, and the compiler maps it onto GPU threads.")

    # -----------------------------------------------------------------------
    banner("SUMMARY")
    # -----------------------------------------------------------------------
    print("1. @triton.jit stored the function; Python never ran its body.")
    print("2. add_kernel[grid](...) built a cache key and, on a miss,")
    print("   parsed the source to an AST and compiled it to a GPU binary.")
    print("3. The binary was cached and launched on the GPU.")
    print("4. Python only passed pointers/arguments and started the launch.")


if __name__ == "__main__":
    main()
