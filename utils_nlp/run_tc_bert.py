import papermill as pm
import os

nb_in  = "examples/text_classification/tc_bert_azureml.ipynb"
nb_out = "executed_tc_bert.ipynb"

pm.execute_notebook(
    nb_in,
    nb_out,
    parameters = dict(
        workspace_name = os.getenv("AML_WORKSPACE"),
        compute_name   = os.getenv("AML_COMPUTE"),      # now cpu‑cluster
        subscription_id= os.getenv("AML_SUBSCRIPTION_ID"),
        resource_group = os.getenv("AML_RESOURCE_GROUP"),
    ),
)
print(f"✓ Notebook finished ‑ results saved to {nb_out}")
