import pickle
import os

def save_model(name:str, model):
    """
    save_model
    -----------
    
    A function to save a previously created model, the output is a `.pkl` file.

    Parameters
    ----------
    
    name: string
        The name of the model to be saved, input a string to save it.
        
    model
        The trained model to be saved.
        
    Usage
    ------
    ```python
    >>> from ocean.save_load import save_model
    >>> save_model("model", model_train)
    ```
    """
    
    if not os.path.exists("save_model"):
        os.makedirs("save_model")
    
    with open(f"save_model/{name}.pkl", "wb") as file:
        pickle.dump(model, file)
