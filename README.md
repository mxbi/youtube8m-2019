# youtube8m-2019
5th Place Solution to 3rd YouTube-8M Video Understanding Challenge (Last Top GB Model)

The code is being finalized. Please stay tuned for final version.

## Running the solution

Our solution was split into two separate branches (a deep learning approach and an XGBoost approach), which were ensembled at the end by rank average.

To run each respective part, see the READMEs in `deep_learning_for_event_localization` and `gbm_for_event_localization`.

### Ensembling submissions

To ensemble two kaggle submissions using a rank average (as in our solution):

```bash
python ensemble.py [submission1.csv[.gz]] [submission2.csv[.gz]] [output_submission.csv[.gz]]
```