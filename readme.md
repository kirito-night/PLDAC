## intro
resource for store initial videos
extract for extract frames in videos
result for store images and databases obtained in tracings

## method1: steady frames analysis
### 1 corner detection
code_extraction:
data_extract.ipynb to extract frames and store in extract files
extract_fourmis.ipynb to locate each ant
extract_fourmis1.ipynb to locate ants massively

### cnn count:
recognition_img_fix:
cnn_generate.ipynb to create train/test set according to 14 imgs
Generate.ipynb rotate images for more training info
cnn_train to define the cnn model, and stock the model im model_frac.pkl (metrics of cnn also in it)
Preprocessing parameters stocked in mu.npy and sig.npy
test_mass.ipynb to see whether the pre-trained model works
nid_count.py to output the result in video

## method2: differential to find moving objects
recognition_opencv;
open_cv.py to find moving ants and store their ids and locations in result in csv
Track.py simple algo1 real time dict


find_trace:
distinct.ipynb  a try to combine the traces, but failed
distinct_pandas.ipynb combine the traces, and store the extracted data in result
find_trace.py show the results in video