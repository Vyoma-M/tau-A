# tau-A 
This package contains the analysis tools for processing of observations of the Crab nebula (also known as Tau-A or M1) by Planck (microwave) and IXPE (X-ray) satellites. Specifically:
1) Procurement of NPIPE Time-ordered Data (TOD)
2) Implementation of mapmaking methods: libmadam wrapper, binning in HealPix pixelization scheme, binning onto a square grid with user-defined number of pixels/pixel size
3) Computation of Unit Conversion (UC) and Colour Correction (CC) for a power-law SED and implementation of these corrections to NPIPE TOD using Planck RIMO files.
4) Conversions between different coordinate systems.
5) Visualisation tools to display Stokes I, Q, U maps of the object and an option to overplot polarization vectors.

![Static Badge](https://img.shields.io/badge/GitHub-Vyoma--M-blue?link=https%3A%2F%2Fgithub.com%2FVyoma-M)


## Acknowledgement
If you found this package useful for your work, please cite my thesis:
[1] Vyoma Muralidhara (2024): Spectral distortion and polarization of the cosmic microwave background: Measurement, challenges and perspectives, [DOI:10.5282/edoc.34768](https://arxiv.org/abs/2503.08538). 
BibTex entry:
```
@phdthesis{Muralidhara:2024hey,
    author = "Muralidhara, Vyoma",
    title = "{Spectral distortion and polarization of the cosmic microwave background}",
    eprint = "2503.08538",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.5282/edoc.34768",
    school = "Munich U.",
    year = "2024"
}
```

## License

Copyright 2026 Vyoma Muralidhara.

tau-A is free software made available under the MIT License. For details see the LICENSE file.
