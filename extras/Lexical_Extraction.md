To avoid code errors caused by inconsistent data formats, you are required to preprocess the target data into JSON format. Each entry in the JSON file should follow the structure below:

```json
{
  "name": "<unique sample ID>",
  "pose_path": "<file path to the corresponding keypoint file>",
  "text": "<textual translation of the sign language video>"
}
```

Field Descriptions:

**name**: The unique identifier for each sample.
**pose_path**: The storage path to the keypoint file associated with this sample.
**text**: The translated text corresponding to the sign language video for this sample.

Then, run the `leixcal_extraction.py` file to get the final dataset file, which can bve directly put into the `data` folder:
```bash
python ./extras/leixcal_extraction.py
            --input_json {your input json file}
            --output_path {the output folder}
            --phase {the phase of the data in the dataset, should be train/dev/test}
            --llada_tokenizer {the LLaDA weights folder (for tokenization)}
            --dataset {dataset name like csl_daily, for naming only}
```