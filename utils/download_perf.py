import argparse
import sys
import os

curl_v1 = "https://storage.live.com/downloadfiles/V1/Zip?&authkey=AM3Xv7w16oqDDv0&application=1141147648"
data_raw_format_v1 = "resIds=9C85B2C346F440CF%{}&canary=YcSqFbuNcf7ZJ8hEk3EWehB7amZnmUeDmCaKX9bO%2FeQ%3D2&authkey=AM3Xv7w16oqDDv0"

# <iframe src="https://onedrive.live.com/embed?cid=9C85B2C346F440CF&resid=9C85B2C346F440CF%21111" width="165" height="128" frameborder="0" scrolling="no"></iframe>

curl_v2 = "https://storage.live.com/downloadfiles/V1/Zip?authKey=%21ACaUbVBSIuDvCrI&application=1141147648"
data_raw_format_v2 = "resIds=60A1A318FA7A3606%{}&canary=LerJFOBG2LJm%2FTP%2BoThDzUjrn%2BnHeGoiRiam4wV0IpA%3D8&authkey=%21ACaUbVBSIuDvCrI"

def download(args): 

    assert args.dataset in ["co3d", "scannet"]

    if args.dataset == "co3d": 
        if args.chunks is None:
            chunks = [str(i).zfill(2) for i in range(100)]
        else:
            chunks = args.chunks.lstrip("[").rstrip("]").split(",")

        for chunk in chunks: 
            chunk = chunk.zfill(2)
            chunk_int = int(chunk)
            outpath = os.path.join(args.outdir, chunk + ".zip")
            if chunk_int > 75:
                data_raw = data_raw_format_v2.format(str(211419 - 76 + chunk_int))
                curl = curl_v2
            else:
                data_raw = data_raw_format_v1.format(str(21111 - 00 + chunk_int))
                curl = curl_v1      

            run_str = f"curl -L \"{curl}\" --data-raw \"{data_raw}\" --compressed --output {outpath}"
            print("Running:", run_str)
            os.system(run_str)

if __name__ == "__main__": 

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices = ["co3d", "scannet"]
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=None
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="."
    )
    args = parser.parse_args()

    download(args)