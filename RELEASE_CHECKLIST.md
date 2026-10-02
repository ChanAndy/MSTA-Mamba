# Release checklist

Items that require author confirmation before making the repository public:

- [ ] Replace `OWNER` in `README.md` and `CITATION.cff` with the GitHub organization/user.
- [ ] Confirm the official BIBM year, proceedings title, DOI, and page numbers.
- [ ] Confirm all author names and ordering in `CITATION.cff`.
- [ ] Confirm that population SD (`ddof=0`) matches the submitted evaluation code.
- [ ] Add the exact EchoNet-Dynamic manifest preparation script after checking the
      downloaded dataset's metadata layout and indexing convention.
- [ ] Run the full EchoNet-Dynamic test split and compare ED/ES MAE with Table I.
- [ ] Select and privacy-review the final POCUS and EchoNet checkpoints.
- [ ] Publish weights outside Git and add URLs plus SHA256 hashes.
- [ ] Confirm the MambaVision code and pretrained-weight redistribution terms.
- [ ] Confirm institutional approval for every public POCUS-derived artifact.
- [ ] Test installation, CPU smoke tests, and one-GPU inference in a fresh environment.
- [ ] Create a GitHub Actions workflow after pinning the supported PyTorch matrix.

