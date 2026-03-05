# Types of missing data

| Type | Code | Description |
| --------------------- | ------------ | ----------- |
| Missing completely at random | MCAR | The reason why the value of that variable is missing is completely at a random circumstances |
| Missing at random | MAR | The reason why the value of that variable is missing is related to the other variable in the dataset |
| Missing not at random | MNAR | The reason why the value of that variable is misisng is because of the variable itself (Hidden, Sensitive, or because of value itself) |

---

Imputation strategies:

- Listwise deletion (Low proportion of missing data).
- Deletion of the entire column (ML cases when the ML Model does not care about that variable).
- Fixed value.
- Mean, median, or mode.
- Forward or backward fill (Works well in time-series or ordered data).
- Hot deck imputation (Look for a record in the dataset with the similar characteristics and fill the value based on that).
- Cold deck imputation (Look for a record in another dataset with the similar characteristics and fill the value based on that).
- Machine learning model.

---

- Questions to ask to determine which imputation strategies to use:
    - How important is the missing data?
        - Not important
            - Listwise deletion or deletion of the entire column
        - If the the missing data is huge proportion and it is important then perform imputation
    - How much data is missing?
    - Do I know why the data is missing?

---

- TODO:
    - Let the Claude check the output and check if it's good now before moving to the next analysis