# -*- coding: utf-8 -*-

"""Main module."""

import numpy as np
import traitlets as tl
import traittypes as tt


class PositiveInt(tl.Int):

    def info(self):
        return u'a positive integer'

    def validate(self, obj, proposal):
        super().validate(obj, proposal)
        if proposal <= 0:
            self.error(obj, proposal)
        return proposal


class PositiveFloat(tl.Float):

    def info(self):
        return u'a positive float'

    def validate(self, obj, proposal):
        super().validate(obj, proposal)
        if proposal <= 0.0:
            self.error(obj, proposal)
        return proposal


class BCC(tl.HasTraits):
    ndim = PositiveInt()
    sizes = tt.Array(dtype='i8')
    lower = tt.Array(dtype='f8')
    upper = tt.Array(dtype='f8')
    width = tt.Array(dtype='f8')

    @tl.default('ndim')
    def _default_ndim(self):
        if len(self.sizes.shape) is not 1 or self.sizes.shape[0] < 3:
            raise tl.TraitError('sizes must have shape (ndim>=3,)')
        return self.sizes.shape[0]

    @tl.default('sizes')
    def _default_sizes(self):
        raise tl.TraitError('sizes must be specified')

    @tl.default('lower')
    def _default_lower(self):
        return np.zeros((self.ndim,))

    @tl.default('upper')
    def _default_upper(self):
        return np.ones((self.ndim,))

    @tl.default('width')
    def _default_width(self):
        return (self.upper - self.lower) / self.sizes

    @tl.validate('lower')
    def _validate_lower(self, proposal):
        print('_validate_lower:', proposal)
        lower = proposal['value']
        if (self.sizes.shape != lower.shape):
            err = 'sizes and lower must have same shape. '
            err += 'sizes.shape: %s, lower.shape: %s' % (
                self.sizes.shape, lower.shape)
            raise tl.TraitError(err)
        return lower

    @tl.validate('upper')
    def _validate_upper(self, proposal):
        upper = proposal['value']
        if (self.sizes.shape != upper.shape):
            err = 'sizes and upper must have same shape. '
            err += 'sizes.shape: %s, upper.shape: %s' % (
                self.sizes.shape, upper.shape)
            raise tl.TraitError(err)
        return upper

    def __len__(self):
        return np.prod(self.sizes) * 2

    def get_bin_indices(self, value):
        value = np.asarray(value)
        tmp = value.copy()
        tmp -= self.lower
        tmp /= self.width
        indices = tmp.astype('i8')
        tmp -= indices + 0.5
        corner_indices = indices - (tmp < 0)
        odd = 0.25 * self.ndim < np.sum(np.sign(tmp) * tmp, axis=-1)
        indices = np.where(odd[..., np.newaxis], corner_indices, indices)
        return indices, odd

    def sizes_prod(self):
        sizes_prod = np.ones(self.ndim, dtype='i8')
        sizes_prod[1:] = np.cumprod(self.sizes[:-1])
        return sizes_prod

    def get_bin_index(self, value):
        indices, odd = self.get_bin_indices(value)
        index = np.sum(self.sizes_prod() * indices, axis=-1)
        return np.left_shift(index, 1) + odd

    def get_bin_center_from_indices(self, indices, odd):
        cen = (self.lower + self.width / 2 + self.width * indices
               + np.where(odd[..., np.newaxis], self.width / 2.0, 0.0))
        return cen

    def get_bin_center(self, index):
        index = np.asarray(index, dtype='i8')
        odd = np.bitwise_and(index, 1)
        indices = np.mod(np.right_shift(index, 1)[..., np.newaxis] //
                         self.sizes_prod(), self.sizes)
        return self.get_bin_center_from_indices(indices, odd)
