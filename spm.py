import torch
import numpy as np
from PIL import Image

def peano_curve_indices(num_patches_h, num_patches_w):
    # Simple Peano curve-like ordering
    indices = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            if i % 2 == 0:
                indices.append((i, j))
            else:
                indices.append((i, num_patches_w - 1 - j))
    return np.array(indices).reshape(-1, 2)

def create_feather_mask(height, width, feather_size=4):
    """
    Create a 2D mask of shape (height x width) that smoothly transitions
    from 1.0 in the interior to 0.0 at the edges over 'feather_size' pixels.
    """
    mask = np.ones((height, width), dtype=np.float32)
    ramp = np.linspace(0, 1, feather_size, dtype=np.float32)

    # Top fade
    mask[:feather_size, :]    *= ramp[:, None]
    # Bottom fade
    mask[-feather_size:, :]   *= ramp[::-1, None]
    # Left fade
    mask[:, :feather_size]    *= ramp[None, :]
    # Right fade
    mask[:, -feather_size:]   *= ramp[None, ::-1]

    return mask

def edgelogic(i, j, patch_height, patch_width, num_patches_h, num_patches_w, overlap):
    """
    Example 'edgelogic' that extends patch size in the middle,
    but does not exceed (patch_height+2*overlap, patch_width+2*overlap).
    Modify as needed for your scenario.
    """
    # Base top-left (no overlap):
    start_h = i * patch_height
    start_w = j * patch_width
    end_h   = start_h + patch_height
    end_w   = start_w + patch_width

    # If i == 0, we add overlap only at the bottom. If i == last, only top, etc.
    # This is just one possible logic:
    if i == 0:
        end_h += 2 * overlap
    elif i == num_patches_h - 1:
        start_h -= 2 * overlap
    else:
        start_h -= overlap
        end_h   += overlap

    if j == 0:
        end_w += 2 * overlap
    elif j == num_patches_w - 1:
        start_w -= 2 * overlap
    else:
        start_w -= overlap
        end_w   += overlap

    # Make sure we don't go negative or beyond image dimension here if needed

    return start_h, end_h, start_w, end_w

class ShufflePatchMix():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta

    def __call__(self, img):
        if  torch.rand(1) > self.mix_prob:
            return img
    
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

class ShufflePatchMix_all():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0):
        patch_height_options = [14, 28, 56, 112]
        random_number = np.random.randint(1, len(patch_height_options))
        self.patch_height = patch_height_options[random_number]
        self.patch_width = patch_height_options[random_number] 
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta

    def __call__(self, img):
        if  torch.rand(1) > self.mix_prob:
            return img
    
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

class ShufflePatchMix_l():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta

    def __call__(self, img):
        if torch.rand(1) > self.mix_prob:
            return img
    
        img_np = np.array(img)
        h, w, c = img_np.shape
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        N = num_patches_h * num_patches_w

        # local region of shuffling
        self.M = max(num_patches_h, 4)

        # Create a list of patches
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_height
                start_w = j * self.patch_width
                patch = img_np[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width]
                patches.append(patch)

        # Rearrange patches using Peano curve-like scan
        peano_indices = peano_curve_indices(num_patches_h, num_patches_w)
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

class ShufflePatchMix_l_all():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0):
        patch_height_options = [14, 28, 56, 112]
        random_number = np.random.randint(1, len(patch_height_options))
        self.patch_height = patch_height_options[random_number]
        self.patch_width = patch_height_options[random_number] 
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta

    def __call__(self, img):
        if torch.rand(1) > self.mix_prob:
            return img
    
        img_np = np.array(img)
        h, w, c = img_np.shape
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        N = num_patches_h * num_patches_w

        # local region of shuffling
        self.M = max(num_patches_h, 4)

        # Create a list of patches
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h = i * self.patch_height
                start_w = j * self.patch_width
                patch = img_np[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width]
                patches.append(patch)

        # Rearrange patches using Peano curve-like scan
        peano_indices = peano_curve_indices(num_patches_h, num_patches_w)
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

class ShufflePatchMixOverlap():
    def __init__(self, 
                 patch_height, 
                 patch_width, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0, 
                 overlap=4):
        """
        patch_height, patch_width : Base size of each patch (without overlap).
        overlap                  : Overlap in pixels on each edge.
        mix_prob                 : Probability of applying ShufflePatchMix.
        alpha, beta              : Beta distribution parameters (lambda sampling).
        """
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.overlap = int(self.patch_height/7)

    def __call__(self, img):
        # 1) Randomly skip
        if torch.rand(1) > self.mix_prob:
            return img

        # 2) Convert to NumPy float32
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # 3) Compute # of patches in each dimension
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width

        # 4) Precompute a large feather mask for the largest possible patch 
        #    (patch_height + 2*overlap x patch_width + 2*overlap)
        #    We'll slice it down for boundary patches if needed
        feather_mask_full = create_feather_mask(
            self.patch_height + 2*self.overlap,
            self.patch_width  + 2*self.overlap,
            feather_size=4
        )

        # 5) Extract patches
        patches = []
        coords = []
        for i_patch in range(num_patches_h):
            for j_patch in range(num_patches_w):
                start_h, end_h, start_w, end_w = edgelogic(
                    i_patch, j_patch,
                    self.patch_height, self.patch_width,
                    num_patches_h, num_patches_w,
                    self.overlap
                )
                # Clip to image boundaries if needed
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h   = min(h, end_h)
                end_w   = min(w, end_w)

                patch = img_np[start_h:end_h, start_w:end_w]
                patches.append(patch)
                coords.append((start_h, end_h, start_w, end_w))

        # 6) Shuffle & mix patches
        N = len(patches)
        indices = np.random.permutation(N)
        mixed_patches = []
        for i_patch in range(N):
            lam = np.random.beta(self.alpha, self.beta)
            patchA = patches[i_patch]
            patchB = patches[indices[i_patch]]
            mixed_patch = lam * patchA + (1 - lam) * patchB
            mixed_patches.append(mixed_patch)

        # 7) Prepare output & weight arrays for soft blending
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        # 8) Blend each patch with the feather mask
        for i_patch, (sh, eh, sw, ew) in enumerate(coords):
            patch_mixed = mixed_patches[i_patch]
            ph, pw, _ = patch_mixed.shape

            # Extract the corresponding portion of the big feather mask
            # if patch is smaller near boundaries, slice the mask
            mask_2d = feather_mask_full[:ph, :pw]
            
            # Convert mask to 3 channels if needed
            if c == 1:
                mask_3d = mask_2d[..., None]
            else:
                mask_3d = np.repeat(mask_2d[..., None], c, axis=2)

            # Feathered patch
            patch_feathered = patch_mixed * mask_3d

            # Accumulate in output
            output[sh:eh, sw:ew] += patch_feathered
            weight[sh:eh, sw:ew] += mask_2d

        # 9) Final divide by weight -> smooth blend
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]

        final_output = img_np
        final_output[1:-1,1:-1] = output[1:-1,1:-1]
        # 10) Convert back to uint8
        final_output = np.clip(final_output, 0, 255).astype(np.uint8)
        return Image.fromarray(final_output)

class ShufflePatchMixOverlap_all():
    def __init__(self, 
                 patch_height, 
                 patch_width, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0):
        """
        patch_height, patch_width : Base size of each patch (without overlap).
        overlap                  : Overlap in pixels on each edge.
        mix_prob                 : Probability of applying ShufflePatchMix.
        alpha, beta              : Beta distribution parameters (lambda sampling).
        """
        patch_height_options = [14, 28, 56, 112]
        random_number = np.random.randint(1, len(patch_height_options))
        self.patch_height = patch_height_options[random_number]
        self.patch_width = patch_height_options[random_number] 
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.overlap = int(self.patch_height/7)

    def __call__(self, img):
        # 1) Randomly skip
        if torch.rand(1) > self.mix_prob:
            return img

        # 2) Convert to NumPy float32
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # 3) Compute # of patches in each dimension
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width

        # 4) Precompute a large feather mask for the largest possible patch 
        #    (patch_height + 2*overlap x patch_width + 2*overlap)
        #    We'll slice it down for boundary patches if needed
        feather_mask_full = create_feather_mask(
            self.patch_height + 2*self.overlap,
            self.patch_width  + 2*self.overlap,
            feather_size=4
        )

        # 5) Extract patches
        patches = []
        coords = []
        for i_patch in range(num_patches_h):
            for j_patch in range(num_patches_w):
                start_h, end_h, start_w, end_w = edgelogic(
                    i_patch, j_patch,
                    self.patch_height, self.patch_width,
                    num_patches_h, num_patches_w,
                    self.overlap
                )
                # Clip to image boundaries if needed
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h   = min(h, end_h)
                end_w   = min(w, end_w)

                patch = img_np[start_h:end_h, start_w:end_w]
                patches.append(patch)
                coords.append((start_h, end_h, start_w, end_w))

        # 6) Shuffle & mix patches
        N = len(patches)
        indices = np.random.permutation(N)
        mixed_patches = []
        for i_patch in range(N):
            lam = np.random.beta(self.alpha, self.beta)
            patchA = patches[i_patch]
            patchB = patches[indices[i_patch]]
            mixed_patch = lam * patchA + (1 - lam) * patchB
            mixed_patches.append(mixed_patch)

        # 7) Prepare output & weight arrays for soft blending
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        # 8) Blend each patch with the feather mask
        for i_patch, (sh, eh, sw, ew) in enumerate(coords):
            patch_mixed = mixed_patches[i_patch]
            ph, pw, _ = patch_mixed.shape

            # Extract the corresponding portion of the big feather mask
            # if patch is smaller near boundaries, slice the mask
            mask_2d = feather_mask_full[:ph, :pw]
            
            # Convert mask to 3 channels if needed
            if c == 1:
                mask_3d = mask_2d[..., None]
            else:
                mask_3d = np.repeat(mask_2d[..., None], c, axis=2)

            # Feathered patch
            patch_feathered = patch_mixed * mask_3d

            # Accumulate in output
            output[sh:eh, sw:ew] += patch_feathered
            weight[sh:eh, sw:ew] += mask_2d

        # 9) Final divide by weight -> smooth blend
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]

        final_output = img_np
        final_output[1:-1,1:-1] = output[1:-1,1:-1]
        # 10) Convert back to uint8
        final_output = np.clip(final_output, 0, 255).astype(np.uint8)
        return Image.fromarray(final_output)

class ShufflePatchMixOverlap_l():
    def __init__(self, 
                 patch_height, 
                 patch_width, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0, 
                 overlap=4):
        """
        patch_height, patch_width : Base size of each patch (without overlap).
        overlap                  : Overlap in pixels on each edge.
        mix_prob                 : Probability of applying ShufflePatchMix.
        alpha, beta              : Beta distribution parameters (lambda sampling).
        """
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.overlap = int(self.patch_height/7)

    def __call__(self, img):
        # 1) Randomly skip
        if torch.rand(1) > self.mix_prob:
            return img

        # 2) Convert to NumPy float32
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # 3) Compute # of patches in each dimension
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        
        # local region of shuffling
        self.M = max(num_patches_h, 4)

        # 4) Precompute a large feather mask for the largest possible patch 
        #    (patch_height + 2*overlap x patch_width + 2*overlap)
        #    We'll slice it down for boundary patches if needed
        feather_mask_full = create_feather_mask(
            self.patch_height + 2*self.overlap,
            self.patch_width  + 2*self.overlap,
            feather_size=4
        )

        # 5) Extract patches
        patches = []
        coords = []
        for i_patch in range(num_patches_h):
            for j_patch in range(num_patches_w):
                start_h, end_h, start_w, end_w = edgelogic(
                    i_patch, j_patch,
                    self.patch_height, self.patch_width,
                    num_patches_h, num_patches_w,
                    self.overlap
                )
                # Clip to image boundaries if needed
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h   = min(h, end_h)
                end_w   = min(w, end_w)

                patch = img_np[start_h:end_h, start_w:end_w]
                patches.append(patch)
                coords.append((start_h, end_h, start_w, end_w))

        # 6) Shuffle & mix patches
        N = len(patches)
        
        # Rearrange patches using Peano curve-like scan
        peano_indices = peano_curve_indices(num_patches_h, num_patches_w)
        reordered_patches = [patches[i * num_patches_w + j] for i, j in peano_indices]

        # Shuffle M indices at a time
        mixed_patches = []
        for i in range(0, N, self.M):
            end = min(i + self.M, N)
            group_indices = np.random.permutation(end - i) + i
            for j in range(i, end):
                shuffled_index = group_indices[j - i]
                lambda_ = np.random.beta(self.alpha, self.beta)
                # mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * reordered_patches[shuffled_index]
                mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * reordered_patches[shuffled_index]
                mixed_patches.append(mixed_patch)

        # 7) Prepare output & weight arrays for soft blending
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        index = 0

        # 8) Blend each patch with the feather mask
        index = 0
        for (row, col) in peano_indices:
            sh, eh, sw, ew = edgelogic(
                row, col,
                self.patch_height, self.patch_width,
                num_patches_h, num_patches_w,
                self.overlap
            )
            patch_mixed = mixed_patches[index]
            ph, pw, _ = patch_mixed.shape

            # Extract the corresponding portion of the big feather mask
            # if patch is smaller near boundaries, slice the mask
            mask_2d = feather_mask_full[:ph, :pw]
            
            # Convert mask to 3 channels if needed
            if c == 1:
                mask_3d = mask_2d[..., None]
            else:
                mask_3d = np.repeat(mask_2d[..., None], c, axis=2)

            # Feathered patch
            patch_feathered = patch_mixed * mask_3d

            # Accumulate in output
            output[sh:eh, sw:ew] += patch_feathered
            weight[sh:eh, sw:ew] += mask_2d
            index += 1

        # 9) Final divide by weight -> smooth blend
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]

        final_output = img_np
        final_output[1:-1,1:-1] = output[1:-1,1:-1]
        # 10) Convert back to uint8

        final_output = np.clip(final_output, 0, 255).astype(np.uint8)
        return Image.fromarray(final_output)
    
class ShufflePatchMixOverlap_l_all():
    def __init__(self, 
                 patch_height, 
                 patch_width, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0, 
                 overlap=4):
        """
        patch_height, patch_width : Base size of each patch (without overlap).
        overlap                  : Overlap in pixels on each edge.
        mix_prob                 : Probability of applying ShufflePatchMix.
        alpha, beta              : Beta distribution parameters (lambda sampling).
        """
        patch_height_options = [14, 28, 56, 112]
        random_number = np.random.randint(1, len(patch_height_options))
        self.patch_height = patch_height_options[random_number]
        self.patch_width = patch_height_options[random_number] 
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.overlap = int(self.patch_height/7)

    def __call__(self, img):
        # 1) Randomly skip
        if torch.rand(1) > self.mix_prob:
            return img

        # 2) Convert to NumPy float32
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # 3) Compute # of patches in each dimension
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        
        # local region of shuffling
        self.M = max(num_patches_h, 4)

        # 4) Precompute a large feather mask for the largest possible patch 
        #    (patch_height + 2*overlap x patch_width + 2*overlap)
        #    We'll slice it down for boundary patches if needed
        feather_mask_full = create_feather_mask(
            self.patch_height + 2*self.overlap,
            self.patch_width  + 2*self.overlap,
            feather_size=4
        )

        # 5) Extract patches
        patches = []
        coords = []
        for i_patch in range(num_patches_h):
            for j_patch in range(num_patches_w):
                start_h, end_h, start_w, end_w = edgelogic(
                    i_patch, j_patch,
                    self.patch_height, self.patch_width,
                    num_patches_h, num_patches_w,
                    self.overlap
                )
                # Clip to image boundaries if needed
                start_h = max(0, start_h)
                start_w = max(0, start_w)
                end_h   = min(h, end_h)
                end_w   = min(w, end_w)

                patch = img_np[start_h:end_h, start_w:end_w]
                patches.append(patch)
                coords.append((start_h, end_h, start_w, end_w))

        # 6) Shuffle & mix patches
        N = len(patches)
        
        # Rearrange patches using Peano curve-like scan
        peano_indices = peano_curve_indices(num_patches_h, num_patches_w)
        reordered_patches = [patches[i * num_patches_w + j] for i, j in peano_indices]

        # Shuffle M indices at a time
        mixed_patches = []
        for i in range(0, N, self.M):
            end = min(i + self.M, N)
            group_indices = np.random.permutation(end - i) + i
            for j in range(i, end):
                shuffled_index = group_indices[j - i]
                lambda_ = np.random.beta(self.alpha, self.beta)
                # mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * reordered_patches[shuffled_index]
                mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * reordered_patches[shuffled_index]
                mixed_patches.append(mixed_patch)

        # 7) Prepare output & weight arrays for soft blending
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)
        index = 0

        # 8) Blend each patch with the feather mask
        index = 0
        for (row, col) in peano_indices:
            sh, eh, sw, ew = edgelogic(
                row, col,
                self.patch_height, self.patch_width,
                num_patches_h, num_patches_w,
                self.overlap
            )
            patch_mixed = mixed_patches[index]
            ph, pw, _ = patch_mixed.shape

            # Extract the corresponding portion of the big feather mask
            # if patch is smaller near boundaries, slice the mask
            mask_2d = feather_mask_full[:ph, :pw]
            
            # Convert mask to 3 channels if needed
            if c == 1:
                mask_3d = mask_2d[..., None]
            else:
                mask_3d = np.repeat(mask_2d[..., None], c, axis=2)

            # Feathered patch
            patch_feathered = patch_mixed * mask_3d

            # Accumulate in output
            output[sh:eh, sw:ew] += patch_feathered
            weight[sh:eh, sw:ew] += mask_2d
            index += 1

        # 9) Final divide by weight -> smooth blend
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]

        final_output = img_np
        final_output[1:-1,1:-1] = output[1:-1,1:-1]
        # 10) Convert back to uint8

        final_output = np.clip(final_output, 0, 255).astype(np.uint8)
        return Image.fromarray(final_output)
    
class JigsawPuzzle():
    def __init__(self, patch_height, patch_width):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.mix_prob = 1

    def __call__(self, img):
        if  torch.rand(1) <= self.mix_prob:
            return img
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
            shuffled_index = indices[i]
            mixed_patch = patches[shuffled_index]
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
    
class JigsawPuzzle_all():
    def __init__(self, patch_height, patch_width):
        patch_height_options = [14, 28, 56, 112]
        random_number = np.random.randint(1, len(patch_height_options))
        self.patch_height = patch_height_options[random_number]
        self.patch_width = patch_height_options[random_number] 
        self.mix_prob = 1

    def __call__(self, img):
        if  torch.rand(1) <= self.mix_prob:
            return img
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
            shuffled_index = indices[i]
            mixed_patch = patches[shuffled_index]
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

class JigsawPuzzle_l():
    def __init__(self, patch_height, patch_width):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.mix_prob = 1

    def __call__(self, img):
        if torch.rand(1) <= self.mix_prob:
            img_np = np.array(img)
            h, w, c = img_np.shape
            num_patches_h = h // self.patch_height
            num_patches_w = w // self.patch_width
            N = num_patches_h * num_patches_w

            # local region of shuffling
            self.M = max(num_patches_h, 4)

            # Create a list of patches
            patches = []
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    start_h = i * self.patch_height
                    start_w = j * self.patch_width
                    patch = img_np[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width]
                    patches.append(patch)

            # Rearrange patches using Peano curve-like scan
            peano_indices = peano_curve_indices(num_patches_h, num_patches_w)
            reordered_patches = [patches[i * num_patches_w + j] for i, j in peano_indices]

            # Shuffle M indices at a time
            mixed_patches = []
            for i in range(0, N, self.M):
                end = min(i + self.M, N)
                group_indices = np.random.permutation(end - i) + i
                for j in range(i, end):
                    shuffled_index = group_indices[j - i]
                    mixed_patch = reordered_patches[shuffled_index]
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


'''

class ShufflePatchMix_Blur():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0, kernel_size=3):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.kernel_size = kernel_size
        self.feather_size = int((patch_height-1)/3)
        self.feather_mask = create_feather_mask(patch_height, patch_width, self.feather_size)


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
            patches_img = []
            mixed_patches_img = []

            # Apply PatchMix transformation
            for i in range(N):
                # Mixing using Lambda sampled from a Beta distribution
                lambda_ = np.random.beta(self.alpha, self.beta)
                shuffled_index = indices[i]
                # patch_blur = cv2.GaussianBlur(, (self.kernel_size, self.kernel_size), 0)
                mask_3d = self.feather_mask[..., None]
                patch_img = (lambda_) * patches[i]
                mixed_patch_img = (1-lambda_) * patches[shuffled_index]
                patches_img.append(patch_img)
                mixed_patches_img.append(mixed_patch_img)

            # Reconstruct the mixed image from mixed patches
            mixed_img = np.zeros_like(img_np)
            org_img = np.zeros_like(img_np)
            index = 0
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    start_h = i * self.patch_height
                    start_w = j * self.patch_width
                    org_img[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width] = patches_img[index]
                    mixed_img[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width] = mixed_patches_img[index]
                    index += 1

            # mixed_img = cv2.GaussianBlur(mixed_img, (self.kernel_size, self.kernel_size), 0)
            final_img = org_img + mixed_img
            # final_img = cv2.GaussianBlur(final_img, (self.kernel_size, self.kernel_size), 0)
            final_img = cv2.fastNlMeansDenoisingColored(final_img, 
                                            None, 
                                            h=10, 
                                            hColor=10, 
                                            templateWindowSize=7, 
                                            searchWindowSize=21)


            
            # Convert back to PIL Image
            final_img = np.clip(final_img, 0, 255).astype(np.uint8)
            
            return Image.fromarray(final_img)
        else:
            return img
        
class ShufflePatchMix_Blur_l():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0, M=4, kernel_size=3):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.M = M
        self.kernel_size = kernel_size

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
                    patch_blur = cv2.GaussianBlur(reordered_patches[shuffled_index], (self.kernel_size, self.kernel_size), 0)
                    mixed_patch = lambda_ * reordered_patches[j] + (1 - lambda_) * patch_blur
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
        
            
class JigsawPuzzle:
    def __call__(self, img):
        """
        Divides the input image into 4 parts and randomly shuffles them such that no two quadrants are adjacent
        to their original positions.

        Args:
        img (PIL.Image): Input image.

        Returns:
        PIL.Image: Shuffled image.
        """
        # Convert the image to a numpy array
        img_np = np.array(img)
        h, w, c = img_np.shape

        # Ensure the image dimensions are divisible by 2
        assert h % 2 == 0 and w % 2 == 0, "Image dimensions must be divisible by 2."

        # Compute midpoints for splitting
        mid_h = h // 2
        mid_w = w // 2

        # Divide the image into 4 parts (quadrants)
        quadrants = [
            img_np[:mid_h, :mid_w],      # Top-left (0)
            img_np[:mid_h, mid_w:],     # Top-right (1)
            img_np[mid_h:, :mid_w],     # Bottom-left (2)
            img_np[mid_h:, mid_w:]      # Bottom-right (3)
        ]

        # Define adjacency constraints
        adjacent_pairs = {
            0: [1, 2],  # Top-left is adjacent to Top-right and Bottom-left
            1: [0, 3],  # Top-right is adjacent to Top-left and Bottom-right
            2: [0, 3],  # Bottom-left is adjacent to Top-left and Bottom-right
            3: [1, 2],  # Bottom-right is adjacent to Top-right and Bottom-left
        }

        # Generate a shuffled order ensuring no adjacency
        N = 4
        while True:
            indices = np.random.permutation(N)
            if not np.array_equal(indices, np.arange(N)):
                if all(indices[i] not in adjacent_pairs[i] for i in range(4)):
                    break

        # Shuffle quadrants using the permuted indices
        shuffled_quadrants = [quadrants[i] for i in indices]

        # Reconstruct the shuffled image
        shuffled_img = np.zeros_like(img_np)
        shuffled_img[:mid_h, :mid_w] = shuffled_quadrants[0]
        shuffled_img[:mid_h, mid_w:] = shuffled_quadrants[1]
        shuffled_img[mid_h:, :mid_w] = shuffled_quadrants[2]
        shuffled_img[mid_h:, mid_w:] = shuffled_quadrants[3]

        # Convert back to a PIL Image
        shuffled_img = np.clip(shuffled_img, 0, 255).astype(np.uint8)
        return Image.fromarray(shuffled_img)



class ShufflePatchMix_New():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0, min_distance_factor=2):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.min_distance_factor = min_distance_factor

    def _is_far_enough(self, idx1, idx2, num_patches_w):
        """Check if two patch indices are far enough apart."""
        row1, col1 = divmod(idx1, num_patches_w)
        row2, col2 = divmod(idx2, num_patches_w)
        return abs(row1 - row2) + abs(col1 - col2) >= self.min_distance

    def __call__(self, img):
        if torch.rand(1) <= self.mix_prob:
            img_np = np.array(img)
            h, w, c = img_np.shape
            num_patches_h = h // self.patch_height
            num_patches_w = w // self.patch_width
            N = num_patches_h * num_patches_w
            self.min_distance = num_patches_h / self.min_distance_factor

            # Create a list of patches
            patches = []
            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    start_h = i * self.patch_height
                    start_w = j * self.patch_width
                    patch = img_np[start_h:start_h+self.patch_height, start_w:start_w+self.patch_width]
                    patches.append(patch)

            # Generate shuffled indices ensuring patches are not too close to their originals
            shuffled_indices = []
            for i in range(N):
                candidates = [j for j in range(N) if self._is_far_enough(i, j, num_patches_w)]
                shuffled_index = np.random.choice(candidates)
                shuffled_indices.append(shuffled_index)

            # Apply PatchMix transformation
            mixed_patches = []
            for i in range(N):
                lambda_ = np.random.beta(self.alpha, self.beta)
                mixed_patch = lambda_ * patches[i] + (1 - lambda_) * patches[shuffled_indices[i]]
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
'''



'''
class ShufflePatchMixOverlap:
    def __init__(self, 
                 patch_height=14, 
                 patch_width=14, 
                 overlap=4, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0):
        """
        patch_height, patch_width: Size of each patch.
        overlap: Number of pixels by which neighboring patches overlap.
        mix_prob: Probability of applying ShufflePatchMix.
        alpha, beta: Parameters for Beta distribution controlling mixing factor.
        """
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.overlap = overlap
        self.mix_prob = mix_prob
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img):
        # Random chance to skip
        if torch.rand(1) > self.mix_prob:
            return img

        # Convert PIL -> float32 NumPy
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # Compute strides (ensure they're not zero or negative)
        stride_h = max(1, self.patch_height - self.overlap)
        stride_w = max(1, self.patch_width  - self.overlap)

        # ----------------------------------------------------
        # 1) Collect Patches in a Single NumPy Array
        # ----------------------------------------------------
        coords_list = []
        row = 0
        while True:
            # Shift up if near bottom
            if row + self.patch_height > h:
                row = max(0, h - self.patch_height)
            done_row = False

            col = 0
            while True:
                # Shift left if near right
                if col + self.patch_width > w:
                    col = max(0, w - self.patch_width)

                coords_list.append((row, col))

                # Move horizontally
                if col + self.patch_width >= w:
                    break
                col += stride_w
                if col >= w:
                    break

            # Move vertically
            if row + self.patch_height >= h:
                done_row = True
            row += stride_h
            if row >= h or done_row:
                break

        # Convert coords_list to a NumPy array of shape (N, 2)
        coords_array = np.array(coords_list, dtype=np.int32)
        N = coords_array.shape[0]

        # Allocate a single array for patches: (N, patch_height, patch_width, c)
        patches_arr = np.zeros((N, self.patch_height, self.patch_width, c), dtype=np.float32)

        # Extract patches in one pass
        for i in range(N):
            top, left = coords_array[i]
            patches_arr[i] = img_np[top : top + self.patch_height, 
                                    left : left + self.patch_width]

        # ----------------------------------------------------
        # 2) Shuffle Indices & Mix in a Single Pass
        # ----------------------------------------------------
        indices = np.random.permutation(N)

        # We'll create an array of lambdas, one for each patch
        lambdas = np.random.beta(self.alpha, self.beta, size=N).astype(np.float32)

        # Create array for mixed patches
        mixed_patches = np.empty_like(patches_arr, dtype=np.float32)

        # Mix each patch with the shuffled partner
        for i in range(N):
            lam = lambdas[i]
            # Weighted sum
            mixed_patches[i] = lam * patches_arr[i] + (1 - lam) * patches_arr[indices[i]]

        # ----------------------------------------------------
        # 3) Reconstruct with Weighted Overlap
        # ----------------------------------------------------
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        # Single loop to blend all patches
        for i in range(N):
            top, left = coords_array[i]
            ph_slice = slice(top, top + self.patch_height)
            pw_slice = slice(left, left + self.patch_width)

            output[ph_slice, pw_slice] += mixed_patches[i]
            weight[ph_slice, pw_slice] += 1.0

        # Avoid division by zero
        np.maximum(weight, 1e-8, out=weight)  # faster than clip in many cases

        # Broadcast divide by weight for each channel
        output /= weight[..., None]

        # Convert back to uint8
        output = np.clip(output, 0, 255).astype(np.uint8)
        return Image.fromarray(output)

class ShufflePatchMixOverlap_l:
    def __init__(self, 
                 patch_height=14, 
                 patch_width=14, 
                 overlap=4, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0, 
                 M=4):
        """
        patch_height, patch_width : Base size of each patch.
        overlap                  : Number of pixels by which neighboring patches overlap.
        mix_prob                 : Probability of applying ShufflePatchMix.
        alpha, beta              : Beta distribution parameters controlling mixing factor.
        M                        : Shuffle locally in groups of size M.
        """
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.overlap = overlap
        self.mix_prob = mix_prob
        self.alpha = alpha
        self.beta = beta
        self.M = M

    def _peano_curve_indices(self, grid_h, grid_w):
        """
        Creates a zigzag (Peano-like) ordering for a grid of size (grid_h x grid_w).
        Returns an array of shape (grid_h*grid_w, 2) with (row, col) for each grid cell
        in peano/zigzag order.
        """
        coords = []
        for row in range(grid_h):
            if row % 2 == 0:  # left to right
                for col in range(grid_w):
                    coords.append((row, col))
            else:             # right to left
                for col in range(grid_w - 1, -1, -1):
                    coords.append((row, col))
        return np.array(coords, dtype=np.int32)

    def __call__(self, img):
        # Random chance to skip
        if torch.rand(1) > self.mix_prob:
            return img

        # Convert PIL -> float32 NumPy
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # Compute strides
        stride_h = max(1, self.patch_height - self.overlap)
        stride_w = max(1, self.patch_width  - self.overlap)

        # -------------------------------------------------------------------
        # (1) Collect patch coordinates so the entire image is covered,
        #     including bottom/right boundaries (shifting if needed).
        # -------------------------------------------------------------------
        coords_list = []
        row = 0
        while True:
            if row + self.patch_height > h:
                row = max(0, h - self.patch_height)
            done_row = False

            col = 0
            while True:
                if col + self.patch_width > w:
                    col = max(0, w - self.patch_width)

                coords_list.append((row, col))

                if col + self.patch_width >= w:
                    break
                col += stride_w
                if col >= w:
                    break

            if row + self.patch_height >= h:
                done_row = True
            row += stride_h
            if row >= h or done_row:
                break

        # Convert coords_list -> NumPy array (N, 2)
        coords_array = np.array(coords_list, dtype=np.int32)
        N = coords_array.shape[0]

        # -------------------------------------------------------------------
        # (2) Extract patches into a single 4D array: (N, ph, pw, c)
        # -------------------------------------------------------------------
        patches_arr = np.zeros((N, self.patch_height, self.patch_width, c), 
                               dtype=np.float32)

        for i in range(N):
            top, left = coords_array[i]
            patches_arr[i] = img_np[top: top + self.patch_height,
                                    left: left + self.patch_width]

        # -------------------------------------------------------------------
        # (3) Build a grid mapping so we know each patch's "grid row" (gh)
        #     and "grid col" (gw). We'll get unique rows, unique cols,
        #     then map (row, col) -> (gh, gw).
        # -------------------------------------------------------------------
        all_rows = coords_array[:, 0]
        all_cols = coords_array[:, 1]
        unique_rows = np.unique(all_rows)
        unique_cols = np.unique(all_cols)
        # Build dict (row_value -> index_in_grid)
        row_idx_map = {val: i for i, val in enumerate(unique_rows)}
        col_idx_map = {val: i for i, val in enumerate(unique_cols)}

        # grid_h, grid_w
        grid_h = len(unique_rows)
        grid_w = len(unique_cols)

        # For each patch i, find (gh, gw)
        grid_map = np.zeros((N, 2), dtype=np.int32)
        for i in range(N):
            gh = row_idx_map[all_rows[i]]
            gw = col_idx_map[all_cols[i]]
            grid_map[i] = (gh, gw)

        # -------------------------------------------------------------------
        # (4) Build the Peano (zigzag) ordering of size (grid_h*grid_w).
        #     Then we create a mapping from (gh,gw) -> index in "peano order".
        # -------------------------------------------------------------------
        peano_coords = self._peano_curve_indices(grid_h, grid_w)  # shape (grid_h*grid_w, 2)

        # We'll invert that to get an array "peano_order" of length N
        # so that peano_order[k] = i means the k-th patch in peano order is patch i.
        # Steps:
        #   1) Build dict from (gh,gw) -> k (the index in peano_coords)
        #   2) For each patch i, grid_map[i] = (gh,gw) -> k
        #   3) Store i in "inv_peano[k] = i"
        inv_dict = {}
        for k, (gh, gw) in enumerate(peano_coords):
            inv_dict[(gh, gw)] = k

        peano_order = np.zeros(N, dtype=np.int32)
        for i in range(N):
            gh, gw = grid_map[i]
            k = inv_dict[(gh, gw)]
            peano_order[k] = i

        # -------------------------------------------------------------------
        # (5) Reorder patches in Peano order, do local scuffling in chunks of M
        # -------------------------------------------------------------------
        # Reorder patches_arr according to peano_order
        # peano_arr[k] = patches_arr[peano_order[k]]
        peano_arr = patches_arr[peano_order]

        # We'll build a new array "mixed_peano_arr" for the scuffled result
        mixed_peano_arr = np.empty_like(peano_arr, dtype=np.float32)

        # Local scuffling
        # Iterate in chunks of size M
        i_start = 0
        while i_start < N:
            i_end = min(i_start + self.M, N)
            local_ids = np.arange(i_start, i_end)
            np.random.shuffle(local_ids)  # random permutation of [i_start..i_end)
            
            # For each index in [i_start..i_end), mix with a random partner in that chunk
            for idx_orig in range(i_start, i_end):
                idx_shuffled = local_ids[idx_orig - i_start]
                lam = np.random.beta(self.alpha, self.beta)

                patchA = peano_arr[idx_orig]
                patchB = peano_arr[idx_shuffled]
                mixed_peano_arr[idx_orig] = lam * patchA + (1 - lam) * patchB
            
            i_start = i_end

        # Now we have "mixed_peano_arr" in Peano order. 
        # We put it back to the original indexing:
        #   patches_arr[peano_order[k]] = mixed_peano_arr[k]
        patches_arr[peano_order] = mixed_peano_arr

        # -------------------------------------------------------------------
        # (6) Reconstruct the final image with weighted blending
        # -------------------------------------------------------------------
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        for i in range(N):
            top, left = coords_array[i]
            patch = patches_arr[i]
            ph, pw, _ = patch.shape
            output[top : top + ph, left : left + pw] += patch
            weight[top : top + ph, left : left + pw] += 1.0

        # Weighted average
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]

        # Convert to uint8
        output = np.clip(output, 0, 255).astype(np.uint8)
        return Image.fromarray(output)

        
class ShufflePatchMixOverlap:
    def __init__(self, 
                 patch_height=14, 
                 patch_width=14, 
                 overlap=4, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0):
        """
        patch_height, patch_width : Size of each patch.
        overlap                  : Number of pixels by which neighboring patches overlap.
        mix_prob                 : Probability of applying ShufflePatchMix.
        alpha, beta              : Parameters for Beta distribution controlling mixing factor.
        """
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.overlap = overlap
        self.mix_prob = mix_prob
        self.alpha = alpha
        self.beta = beta

    def __call__(self, img):
        if torch.rand(1) > self.mix_prob:
            return img
        
        # Convert PIL -> NumPy float32
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # Compute the stride for each dimension
        # e.g., if patch_width=32 and overlap=8, then stride_w=24
        stride_h = self.patch_height - self.overlap
        stride_w = self.patch_width - self.overlap
        stride_h = max(1, stride_h)  # Avoid zero or negative
        stride_w = max(1, stride_w)

        patches = []
        coords = []

        # Slide over the image, ensuring full coverage (including boundaries)
        row = 0
        while True:
            # If the bottom of this patch goes past the image, shift up so it exactly fits
            if row + self.patch_height > h:
                row = h - self.patch_height
            if row < 0:
                row = 0

            col = 0
            done_row = False

            while True:
                # If the right edge of this patch goes past the image, shift left so it exactly fits
                if col + self.patch_width > w:
                    col = w - self.patch_width
                if col < 0:
                    col = 0

                # Extract patch
                patch = img_np[row : row + self.patch_height,
                               col : col + self.patch_width]
                patches.append(patch)
                coords.append((row, col))

                # Move to the next column
                if col + self.patch_width >= w:
                    break  # Reached the right boundary
                col += stride_w
                if col >= w:
                    break

            # Move to the next row
            if row + self.patch_height >= h:
                done_row = True
            row += stride_h
            if row >= h or done_row:
                break

        # Shuffle indices
        N = len(patches)
        indices = np.random.permutation(N)

        # Apply PatchMix transformation
        mixed_patches = []
        for i in range(N):
            lam = np.random.beta(self.alpha, self.beta)
            patchA = patches[i]
            patchB = patches[indices[i]]
            # Mix the two patches (they should be the same shape here)
            mixed_patch = lam * patchA + (1 - lam) * patchB
            mixed_patches.append(mixed_patch)

        # Reconstruct with blending in overlapped areas
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        for mixed_patch, (top, left) in zip(mixed_patches, coords):
            ph, pw, _ = mixed_patch.shape
            output[top : top + ph, left : left + pw] += mixed_patch
            weight[top : top + ph, left : left + pw] += 1.0

        # Avoid division by zero
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]  # Broadcast over the color channel

        # Convert back to uint8
        output = np.clip(output, 0, 255).astype(np.uint8)
        return Image.fromarray(output)


class ShufflePatchMixOverlap_l:
    def __init__(self, 
                 patch_height=14, 
                 patch_width=14, 
                 overlap=4, 
                 mix_prob=0.8, 
                 alpha=4.0, 
                 beta=2.0, 
                 M=4):
        """
        patch_height, patch_width : Base size of each patch.
        overlap                  : Number of pixels by which neighboring patches overlap.
        mix_prob                 : Probability of applying ShufflePatchMix.
        alpha, beta              : Beta distribution parameters controlling mixing factor.
        M                        : Shuffle locally in groups of size M.
        """
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.overlap = overlap
        self.mix_prob = mix_prob
        self.alpha = alpha
        self.beta = beta
        self.M = M

    def _peano_curve_indices(self, grid_h, grid_w):
        """
        Creates a zigzag (Peano-like) ordering of patch coordinates.
        Returns an array of shape (grid_h*grid_w, 2) with (row, col) for each patch.
        """
        coords = []
        for row in range(grid_h):
            if row % 2 == 0:
                # left to right
                for col in range(grid_w):
                    coords.append((row, col))
            else:
                # right to left
                for col in range(grid_w - 1, -1, -1):
                    coords.append((row, col))
        return np.array(coords)

    def __call__(self, img):
        # Random chance to skip mixing
        if torch.rand(1) > self.mix_prob:
            return img

        # Convert PIL -> NumPy float32
        img_np = np.array(img, dtype=np.float32)
        h, w, c = img_np.shape

        # Compute stride
        stride_h = self.patch_height - self.overlap
        stride_w = self.patch_width  - self.overlap
        stride_h = max(stride_h, 1)  # avoid zero or negative
        stride_w = max(stride_w, 1)

        # --------------------------------------------------
        # 1) Collect patches so the entire image is covered
        #    including bottom/right boundaries
        # --------------------------------------------------
        patches = []
        coords = []

        row = 0
        while True:
            # If adding patch_height goes beyond bottom, shift up so it fits exactly
            if row + self.patch_height > h:
                row = max(0, h - self.patch_height)

            done_row = False
            col = 0
            while True:
                # If adding patch_width goes beyond right, shift left so it fits exactly
                if col + self.patch_width > w:
                    col = max(0, w - self.patch_width)

                # Extract patch
                patch = img_np[row : row + self.patch_height,
                               col : col + self.patch_width]
                patches.append(patch)
                coords.append((row, col))

                # Move to the next column
                if col + self.patch_width >= w:
                    # We've reached or exceeded the right boundary
                    break
                col += stride_w
                if col >= w:
                    break

            # Move to the next row
            if row + self.patch_height >= h:
                done_row = True
            row += stride_h
            if row >= h or done_row:
                break

        # Total number of patches
        N = len(patches)

        # --------------------------------------------------
        # 2) Reorder patches in a Peano-curve (zigzag) pattern
        # --------------------------------------------------
        # Figure out how many patch-rows & patch-cols we ended up with
        # so we can map them to a (row, col) grid in reading order.
        # Because of the shifting at boundaries, the “effective” grid
        # might have repeated row or col entries at the end. 
        # We'll build a 2D grid by unique positions.
        unique_rows = sorted(list({r for (r, _) in coords}))
        unique_cols = sorted(list({c for (_, c) in coords}))

        # Build a dictionary from (row, col) -> index in a grid
        row_idx_map = {val: i for i, val in enumerate(unique_rows)}
        col_idx_map = {val: i for i, val in enumerate(unique_cols)}
        grid_h = len(unique_rows)
        grid_w = len(unique_cols)

        # We can store patches in a grid array:
        grid = [None] * (grid_h * grid_w)
        for i, (r, c) in enumerate(coords):
            gh = row_idx_map[r]
            gw = col_idx_map[c]
            grid[gh * grid_w + gw] = patches[i]

        # Create a Peano ordering of (gh, gw)
        peano_coords = self._peano_curve_indices(grid_h, grid_w)  # shape (grid_h*grid_w, 2)

        # Build a linear list in the Peano order
        peano_list = []
        for (gh, gw) in peano_coords:
            peano_list.append(grid[gh * grid_w + gw])

        # --------------------------------------------------
        # 3) Local shuffle in groups of size M
        # --------------------------------------------------
        mixed_peano_list = []
        i = 0
        while i < N:
            group_end = min(i + self.M, N)
            # Indices of this group in [i..group_end)
            local_ids = np.arange(i, group_end)
            np.random.shuffle(local_ids)
            
            # For each patch in [i..group_end), mix with a randomly chosen patch in the same group
            for idx_orig in range(i, group_end):
                # pick a partner from local_ids
                idx_shuffled = local_ids[idx_orig - i]
                lam = np.random.beta(self.alpha, self.beta)

                patchA = peano_list[idx_orig]
                patchB = peano_list[idx_shuffled]

                # Mix them
                mixed_patch = lam * patchA + (1 - lam) * patchB
                mixed_peano_list.append(mixed_patch)
            i = group_end

        # --------------------------------------------------
        # 4) Reconstruct image with weighted blending
        # --------------------------------------------------
        # We place the *mixed_peano_list* back into the original “coords” arrangement.
        # But we must do so in the same Peano order => same (gh, gw).
        # Then we map (gh, gw) -> (row, col) actual top-left in the image.
        output = np.zeros_like(img_np, dtype=np.float32)
        weight = np.zeros((h, w), dtype=np.float32)

        # Re-map each (gh, gw) to original top-left
        idx = 0
        for (gh, gw) in peano_coords:
            # Find actual image coords
            row_top = unique_rows[gh]
            col_left = unique_cols[gw]
            patch = mixed_peano_list[idx]
            idx += 1

            ph, pw, _ = patch.shape
            output[row_top : row_top + ph, col_left : col_left + pw] += patch
            weight[row_top : row_top + ph, col_left : col_left + pw] += 1.0

        # Weighted averaging for overlaps
        weight = np.clip(weight, 1e-8, None)
        output /= weight[..., None]  # broadcast over channels

        # Convert to uint8
        output = np.clip(output, 0, 255).astype(np.uint8)
        return Image.fromarray(output)

'''


'''
My code
def create_feather_mask(height, width, feather_size=4):
    """
    Create a 2D mask (height x width) that smoothly transitions 
    from 1 in the center to 0 near edges over 'feather_size' pixels.
    """
    mask = np.ones((height, width), dtype=np.float32)
    ramp = np.linspace(0, 1, feather_size, dtype=np.float32)

    # Fade top
    mask[:feather_size, :]    *= ramp[:, None]
    # Fade bottom
    mask[-feather_size:, :]   *= ramp[::-1, None]
    # Fade left
    mask[:, :feather_size]    *= ramp[None, :]
    # Fade right
    mask[:, -feather_size:]   *= ramp[None, ::-1]

    return mask

def edgelogic(i,j,patch_height,patch_width, num_patches_h,num_patches_w,overlap):
    
    start_h = i * patch_height
    start_w = j * patch_width
    if i==0:
        end_h = start_h + patch_height + 2*overlap
    elif i==num_patches_h-1:
        end_h = start_h + patch_height
        start_h = start_h - 2*overlap
    else:
        end_h = start_h + patch_height + overlap
        start_h = start_h - overlap

    if j==0:
        end_w = start_w + patch_width + 2*overlap
    elif j == num_patches_w-1:
        end_w = start_w + patch_width
        start_w = start_w - 2*overlap
    else:
        end_w = start_w + patch_width + overlap
        start_w = start_w - overlap

    assert((end_h-start_h)==(patch_height + 2*overlap))
    assert((end_w-start_w)==(patch_width + 2*overlap))

    return start_h, end_h, start_w, end_w

class ShufflePatchMixOverlap():
    def __init__(self, patch_height, patch_width, mix_prob=0.8, alpha=4.0, beta=2.0, overlap=2):
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.alpha = alpha
        self.mix_prob = mix_prob
        self.beta = beta
        self.overlap = overlap

    def __call__(self, img):
        if  torch.rand(1) > self.mix_prob:
            return img
    
        img_np = np.array(img)
        
        h, w, c = img_np.shape
        num_patches_h = h // self.patch_height
        num_patches_w = w // self.patch_width
        N = num_patches_h * num_patches_w

        # Create a list of patches
        feather_mask = create_feather_mask(self.patch_height + 2*self.overlap, self.patch_width + 2*self.overlap, feather_size=4)
        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h, end_h, start_w, end_w = edgelogic(i,j,self.patch_height,self.patch_width, num_patches_h,num_patches_w,self.overlap)
                patch = img_np[start_h:end_h, start_w:end_w]
                
                patches.append(patch)

        # Generate a random permutation of the indices
        N = len(patches)
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
        mixed_img = img_np
        index = 0
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                start_h, end_h, start_w, end_w=edgelogic(i,j,self.patch_height,self.patch_width, num_patches_h,num_patches_w,self.overlap)
                mixed_img[start_h:end_h, start_w:end_w] = mixed_patches[index]
                index += 1

        # Convert back to PIL Image
        mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)
        return Image.fromarray(mixed_img)

'''