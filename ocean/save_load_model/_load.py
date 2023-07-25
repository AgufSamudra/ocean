import pickle

def load_model(path:str):
    """
    load_model
    -----------
    
    A function to load a previously saved model, the input is a `.pkl` file.

    Parameters
    ----------
    
    path: string
        The path of the model to be loaded, input a string to load it.
        
    Returns
    -------
    
    model
        The loaded model.
        
    Usage
    ------
    ```python
    >>> from ocean.save_load import load_model
    >>> model = load_model("model")
    ```
    """
    
    with open(f"{path}", "rb") as file:
        model = pickle.load(file)
    
    return model
