import torch
import numpy as np
from PIL import Image


class ShufflePatchMix():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta

    def __call__(self, img):
        if  torch.rand(1) <= self.mix_prob:
            img_np = np.array(img)
            h, w, c = img_np.shape
            num_patches_h = h // self.patch_height
            num_patches_w = w // self.patch_width
            N = num_patches_h * num_patches_w

            # Create a list of patches
            patches = []
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    start_h = i * self.patch_height
                    start_w = j * self.patch_width
                    patch = img_np[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width]
                    patches.append(patch)

            # Generate a random permutation of the indices
            indices = np.random.permutation(N)

            # Initialize the mixed patches list
            mixed_patches = []

            # Apply PatchMix transformation
            for i in range(N):
                # Mixing using Lambda sampled from a Beta distribution
                lambda_ = np.random.beta(self.alpha, self.beta)
                shuffled_index = indices[i]
                mixed_patch = lambda_ * patches[i] + (1 - lambda_) * patches[shuffled_index]
                mixed_patches.append(mixed_patch)

            # Reconstruct the mixed image from mixed patches
            mixed_img = np.zeros_like(img_np)
            index = 0
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    start_h = i * self.patch_height
                    start_w = j * self.patch_width
                    mixed_img[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width] = mixed_patches[index]
                    index += 1

            # Convert back to PIL Image
            mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
            
            return Image.fromarray(mixed_img)
        else:
            return img

    def __repr__(self):
        return f"PatchMixR(patch_height={self.patch_height}, patch_width={self.patch_width}, prob_p={self.mix_prob}, alpha={self.alpha}, beta={self.beta})"



class ShufflePatchMix_l():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0, M=4):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.M = M

    def peano_curve_indices(self, num_patches_h, num_patches_w):
        # Simple Peano curve-like ordering
        indices = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                if i % 2 == 0:
                    indices.append((i, j))
                else:
                    indices.append((i, num_patches_w - 1 - j))
        return np.array(indices).reshape(-1, 2)

    def __call__(self, img):
        if torch.rand(1) <= self.mix_prob:
            img_np = np.array(img)
            h, w, c = img_np.shape
            num_patches_h = h // self.patch_height
            num_patches_w = w // self.patch_width
            N = num_patches_h * num_patches_w

            # Create a list of patches
            patches = []
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    start_h = i * self.patch_height
                    start_w = j * self.patch_width
                    patch = img_np[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width]
                    patches.append(patch)

            # Rearrange patches using Peano curve-like scan
            peano_indices = self.peano_curve_indices(num_patches_h, num_patches_w)
            reordered_patches = [patches[i * num_patches_w + j] for i, j in peano_indices]

            # Shuffle M indices at a time
            mixed_patches = []
            for i in range(0, N, self.M):
                end = min(i + self.M, N)
                group_indices = np.random.permutation(end - i) + i
                for j in range(i, end):
                    shuffled_index = group_indices[j - i]
                    lambda_ = np.random.beta(self.alpha, self.beta)
                    mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * reordered_patches[shuffled_index]
                    mixed_patches.append(mixed_patch)

            # Reconstruct the mixed image from mixed patches by undoing the Peano scan
            mixed_img = np.zeros_like(img_np)
            index = 0
            for i, (row, col) in enumerate(peano_indices):
                start_h = row * self.patch_height
                start_w = col * self.patch_width
                mixed_img[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width] = mixed_patches[index]
                index += 1

            # Convert back to PIL Image
            mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
            return Image.fromarray(mixed_img)
        else:
            return img

    def __repr__(self):
        return (f"PatchMixR(patch_height={self.patch_height}, patch_width={self.patch_width}, "
                f"mix_prob={self.mix_prob}, alpha={self.alpha}, beta={self.beta})")

