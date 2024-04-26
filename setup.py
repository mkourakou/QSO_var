import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="QSO_var",
    version="0.1",
    author="mado kourakou",
    author_email="m.kourakou@noa.gr",
    description="NXSV inference for X-ray AGN lightcurves",
    long_description=long_description,
    packages=["QSO_var"],
    install_requires=["numpy", "matplotlib", "seaborn", "pandas", "scipy", "stan"]
)
