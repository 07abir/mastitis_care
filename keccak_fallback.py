"""Pure-Python Keccak-256, used only when web3 / eth-hash are unavailable.

Ethereum's keccak256 is NOT NIST SHA3-256: the domain-separation padding byte is
0x01 instead of 0x06. Everything else (Keccak-f[1600] permutation, rate 1088,
capacity 512) is identical.

Verified against the canonical vectors:
    keccak256("")        = c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
    keccak256("abc")     = 4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45
    keccak256("hello")   = 1c8aff950685c2ed4bc3174f3472287b56d9517b9c948127319a09a7a36deac8
    keccak256("testing") = 5f16f4c7f149ac4f9510d9cf8cf384038ad348b3bcdc01915f95de12df9d1b02

Run `python -m src.keccak_fallback` to re-check those.
"""

_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A,
    0x8000000080008000, 0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009, 0x000000000000008A,
    0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

# rho rotation offsets, indexed [x][y]
_R = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

_MASK = (1 << 64) - 1
_RATE = 136  # 1088 bits


def _rol(x, n):
    n %= 64
    return ((x << n) | (x >> (64 - n))) & _MASK


def _keccak_f(A):
    for rnd in range(24):
        # theta
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x - 1) % 5] ^ _rol(C[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                A[x][y] ^= D[x]

        # rho + pi
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = _rol(A[x][y], _R[x][y])

        # chi
        for x in range(5):
            for y in range(5):
                A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y] & _MASK)

        # iota
        A[0][0] ^= _RC[rnd]
    return A


def keccak256(data: bytes) -> bytes:
    """Return the 32-byte Keccak-256 digest of ``data``."""
    if isinstance(data, str):
        data = data.encode("utf-8")

    padlen = _RATE - (len(data) % _RATE)
    if padlen == 1:
        padded = data + b"\x81"  # first and last pad byte coincide
    else:
        padded = data + b"\x01" + b"\x00" * (padlen - 2) + b"\x80"

    A = [[0] * 5 for _ in range(5)]
    for off in range(0, len(padded), _RATE):
        block = padded[off:off + _RATE]
        for i in range(_RATE // 8):
            A[i % 5][i // 5] ^= int.from_bytes(block[i * 8:i * 8 + 8], "little")
        A = _keccak_f(A)

    out = b""
    i = 0
    while len(out) < 32:
        out += A[i % 5][i // 5].to_bytes(8, "little")
        i += 1
    return out[:32]


if __name__ == "__main__":
    vectors = [
        (b"", "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
        (b"abc", "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"),
        (b"hello", "1c8aff950685c2ed4bc3174f3472287b56d9517b9c948127319a09a7a36deac8"),
        (b"testing", "5f16f4c7f149ac4f9510d9cf8cf384038ad348b3bcdc01915f95de12df9d1b02"),
    ]
    ok = True
    for data, want in vectors:
        got = keccak256(data).hex()
        ok &= got == want
        print(f"  {'PASS' if got == want else 'FAIL'}  keccak256({data!r}) = {got}")
    print("all vectors passed" if ok else "MISMATCH - do not use this fallback")
