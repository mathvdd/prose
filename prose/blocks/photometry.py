import numpy as np
from astropy.stats import gaussian_sigma_to_fwhm, sigma_clipped_stats
from photutils.aperture import aperture_photometry

from prose import Block, Image

__all__ = ["AperturePhotometry", "AnnulusBackground", "SkyBackground"]


class AperturePhotometry(Block):
    def __init__(self, radii: np.ndarray = None, scale: bool = True, name=None):
        """Perform aperture photometry of each sources.

        |read| :code:`Image.data`, :code:`Image.sources`

        |write| :code:`Image.aperture`

        Parameters
        ----------
        radii : np.ndarray, optional
            apertures radii (definition varies depending on sources), by default None
        scale : bool, optional
            whether to scale radii with :code:`Image.fwhm` usually present in :code:`Image.epsf`, by default True
        name : str, optional
            name of the block, by default None
        """
        super().__init__(name=name, read=["sources", "data"])
        if radii is None:
            # log-uniform
            self._radii = np.exp(np.linspace(np.log(0.1), np.log(12), 30))
        else:
            self._radii = radii
        self.scale = scale

        if self.scale:
            self.read.append("fwhm")

    def run(self, image: Image):
        if self.scale:
            radii = np.array(image.fwhm * self._radii)
        else:
            radii = np.array(self._radii)

        apertures = [image.sources.apertures(r) for r in radii]
        aperture_fluxes = np.array(
            [aperture_photometry(image.data, a)["aperture_sum"].data for a in apertures]
        ).T

        image.aperture = {"fluxes": aperture_fluxes, "radii": radii}

    @property
    def citations(self) -> list:
        return super().citations + ["photutils"]


class _AnnulusPhotometry(Block):
    def __init__(self, name=None, rin=5, rout=8, scale=True):
        super().__init__(name=name, read=["sources", "data"])
        self.rin = rin
        self.rout = rout
        self.scale = scale

    @property
    def citations(self) -> list:
        return super().citations + ["photutils"]


class AnnulusBackground(_AnnulusPhotometry):
    def __init__(
        self,
        rin: float = 5,
        rout: float = 8,
        sigma: float = 3,
        scale=True,
        name: str = None,
    ):
        """Estimate background around each source using an annulus aperture.

        |read| :code:`Image.data`, :code:`Image.sources`

        |write| :code:`Image.annulus`

        Parameters
        ----------
        rin : float, optional
            inner radius of the annulus, by default 5
        rout : float, optional
            outer radius of the annulus, by default 8
        sigma : float, optional
            sigma clipping applied to pixel within annulus before taking the median value, by default 3.
        scale : bool, optional
            whether to scale annulus to EPSF fwhm, by default True. If True, each image must contain an effective PSF and its model (e.g. using :py:class:`~prose.blocks.psf.MedianEPSF` and one of :py:class:`~prose.blocks.psf.Gaussian2D`)
        name : str, optional
            name of the block, by default None
        """
        super().__init__(name=name, rin=rin, rout=rout, scale=scale)
        self.sigma = sigma

    def run(self, image: Image):
        if self.scale:
            fwhm = image.fwhm
            rin = float(fwhm * self.rin)
            rout = float(fwhm * self.rout)
        else:
            rin = self.rin
            rout = self.rout

        annulus = image.sources.annulus(rin, rout)
        annulus_masks = annulus.to_mask(method="center")

        bkg_median = []
        for mask in annulus_masks:
            annulus_data = mask.multiply(image.data)
            if annulus_data is not None:
                annulus_data_1d = annulus_data[mask.data > 0]
                _, median_sigma_clip, _ = sigma_clipped_stats(
                    annulus_data_1d, sigma=self.sigma
                )
                bkg_median.append(median_sigma_clip)
            else:
                bkg_median.append(0.0)

        image.computed["annulus"] = {
            "rin": rin,
            "rout": rin,
            "median": np.array(bkg_median),
            "sigma": self.sigma,
        }


class SkyBackground(Block):
    def __init__(
        self,
        rim: int = 100,
        sigma: float = 3,
        image_subtract: bool = False,
        name=None,
    ):
        """Estimate the sky background and sigma from the edges of the image.

        |read| :code:`Image.data`

        |write| :code:`Image.sky_background`, :code:`Image.data`

        Parameters
        ----------
        rim : int, optional
            size of the rim from which to extract the background value, by default 100
        sigma : float, optional
            sigma clipping applied to pixel the rim before taking the median value, by default 3.
        image_subtract : bool, optional
            whether to subtract the computed sky background from the image, by default False
        name : str, optional
            name of the block, by default None
        """

        super().__init__(name=name)
        self.rim = rim
        self.image_subtract = image_subtract
        self.sigma = sigma

    def run(self, image: Image):

        imshape = image.data.shape
        image_subtract = self.image_subtract

        mask = np.zeros(imshape, dtype = 'bool')
        mask[self.rim:imshape[0]-self.rim,self.rim:imshape[1]-self.rim] = 1

        masked_data = np.ma.masked_array(image.data, mask)

        _, median, std = sigma_clipped_stats(masked_data, sigma=self.sigma)

        if self.image_subtract:
            image.data -= median

        image.computed["sky_background"] = {
            "rim": self.rim,
            "median": median,
            "std": std,
            "sigma": self.sigma,
        }
