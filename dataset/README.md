# Photo-to-Citeseer data

The runnable example uses preprocessed OFA-CS tensor caches. Download `OFA-CS_Photo_Citeseer_Data_20261030.zip` from GitHub Releases and extract it into the repository root.

Required files:

| Relative path | Size | SHA-256 |
|---|---:|---|
| `dataset/photo_d64_khop4_Nl10_pos10_neg10.pt` | 40,297,388 bytes | `8F15CB3868762FBC9D00832D986D2BDFC2FA9963D45CFB8EDF49F06F8186B00C` |
| `dataset/citeseer_d64_khop4_Nl10_pos10_neg10.pt` | 15,260,176 bytes | `E61F04CF3A7F84AA3E8814B3A6B64BCB9B36AF040D44BC30E51CE1F654FEED9E` |
| `dataset/citeseer/citeseer.query` | 1,089 bytes | `1728D64B38F612078907C80FF8D9B906B2F2FC9A3F040C73AD3415814DAFD9F1` |
| `dataset/citeseer/citeseer.gt` | 263,833 bytes | `D9129CAD72765E57CD80C7B5D8C71720B99E699E650AB19BACE10643A0F4C805` |

These four files are sufficient for the released example. The raw Photo and Citeseer graph files are not required because `Dataset` loads the matching preprocessed cache before attempting raw-data preprocessing.
