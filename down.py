import urllib.request

urls = [
  "https://storage.openvinotoolkit.org/repositories/open_model_zoo/temp/human-pose-estimation-0007/FP16/human-pose-estimation-0007.xml",
  "https://storage.openvinotoolkit.org/repositories/open_model_zoo/temp/human-pose-estimation-0007/FP16/human-pose-estimation-0007.bin"
]

for url in urls:
    print(f"Downloading {url.split('/')[-1]} ...")
    urllib.request.urlretrieve(url, url.split('/')[-1])
print("Done!")
