import argparse
from PIL import Image

import numpy as np

np.random.seed(239)


def int_to_uint8_seq(x, size):
    '''
    Represents number as a sequence of bytes (result is an array with values ranged [0; 255])
    '''
    res = []
    for i in range(size):
        res.append(x % 256)
        x //= 256
    return res[::-1]


class SVD():
    def __init__(self, U, S, V):
        self.U = U
        self.S = S
        self.V = V

    @staticmethod
    def numpy(matrix):
        U, S, V = np.linalg.svd(matrix)
        return SVD(U, S, V)

    @staticmethod
    def simple(matrix):
        m_float = matrix.astype('float64')
        M = (m_float @ m_float.T)
        N_ITERS = 50
        n, m = matrix.shape
        S = np.zeros(n)
        U = np.zeros((n, n))
        for i in range(n):
            b = np.random.normal(size=(n, 1))
            for j in range(N_ITERS):
                prod = M @ b
                b = prod / np.linalg.norm(prod)
            eigenvalue = (b.T @ M @ b) / np.linalg.norm(b)
            S[i] = np.sqrt(eigenvalue) if eigenvalue >= 0 else 0
            U[:, i] = b[:, 0]
            M -= (b * eigenvalue) @ b.T
        V = np.diag([1 / x if x != 0 else 0 for x in S]) @ U.T @ m_float
        return SVD(U, S, V)

    @staticmethod
    def advanced(matrix):
        N_ITERS = 50
        n, m = matrix.shape
        U = np.zeros((n, n))
        S = np.zeros(n)
        V = np.zeros((n, m))

        for i in range(N_ITERS):
            Q, R = np.linalg.qr(matrix @ V.T)
            U = Q
            Q, R = np.linalg.qr(matrix.T @ U)
            V = Q.T
            S = np.diag(R[:n, :n])
        
        return SVD(U, S, V)


class CompressedImage():
    FLOAT_IN_BYTES = 8
    RESERVED_BYTES = 6 # (2 + 2 + 2) - for storing n, m, k

    def __init__(self, channels):
        self.channels = channels
    
    @staticmethod
    def compress(matrix, n_times=1, method='numpy'):
        '''
        If initial size is NxM, then compressed size is Kx(N+M+1)
        We estimate K to get: NxM / (Kx(N+M+1)) >= n_times
        '''
        n, m, n_channels = matrix.shape
        k = int(n_channels * n * m / (n_channels * (n + m + 1) * CompressedImage.FLOAT_IN_BYTES + CompressedImage.RESERVED_BYTES) / n_times)
        channels = []
        for i in range(n_channels):
            if method == 'numpy':
                svd = SVD.numpy(matrix[:, :, i])
            elif method == 'simple':
                svd = SVD.simple(matrix[:, :, i])
            elif method == 'advanced':
                svd = SVD.advanced(matrix[:, :, i])
            else:
                raise ValueError(f'method={method} is not supported')
            channels.append(SVD(svd.U[:, :k], svd.S[:k], svd.V[:k, :]))
        return CompressedImage(channels)
    
    def decompress(self):
        matrix = []
        for svd in self.channels:
            matrix.append(svd.U @ np.diag(svd.S) @ svd.V)
        return np.array(matrix).transpose(1, 2, 0).astype('float64')
    
    @staticmethod
    def from_file(filename):
        with open(filename, 'rb') as f:
            n = int.from_bytes(f.read(2), byteorder='big')
            m = int.from_bytes(f.read(2), byteorder='big')
            k = int.from_bytes(f.read(2), byteorder='big')
            data = np.fromfile(f, dtype='float64')
            n_channels = data.size // (k * (n + m + 1))

            channels = []
            pos = 0
            for i in range(n_channels):
                l1 = pos + n * k
                l2 = l1 + k
                l3 = l2 + m * k
                U = data[pos: l1].reshape(n, k)
                S = data[l1: l2]
                V = data[l2: l3].reshape(k, m)
                channels.append(SVD(U, S, V))
                pos = l3
        return CompressedImage(channels)
    
    def to_file(self, filename):
        n = self.channels[0].U.shape[0]
        m = self.channels[0].V.shape[1]
        k = self.channels[0].S.shape[0]

        reserved_data = np.concatenate([
            int_to_uint8_seq(n, 2),
            int_to_uint8_seq(m, 2),
            int_to_uint8_seq(k, 2)
        ]).astype('uint8').tobytes()

        data = np.concatenate([
            np.array([
                np.concatenate([
                    svd.U.reshape(-1),
                    svd.S.reshape(-1),
                    svd.V.reshape(-1)
                ]) for svd in self.channels
            ]).reshape(-1)
        ]).astype('float64').tobytes()

        with open(filename, 'wb') as f:
            f.write(reserved_data + data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['compress', 'decompress'], required=True)
    parser.add_argument('--method', type=str, choices=['numpy', 'simple', 'advanced'])
    parser.add_argument('--compression', type=float, default=1)
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)

    args = parser.parse_args()

    if args.mode == 'compress':
        img = np.array(Image.open(args.in_file))
        compressed_image = CompressedImage.compress(img, args.compression, args.method)
        compressed_image.to_file(args.out_file)
    elif args.mode == 'decompress':
        compressed_image = CompressedImage.from_file(args.in_file)
        img = Image.fromarray(compressed_image.decompress().astype('uint8'))
        img.save(args.out_file)

    else:
        print(f'Error! Mode {args.mode} is incorrect')