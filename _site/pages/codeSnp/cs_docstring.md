#docstring

## Docstring formats
- Google
- Numpydoc

```python
# print out the docstrings
print(the_answer.__doc__)

# or to get cleaner version

import inspect
print(inspect.getdoc(the_answer))

```


### Google Style

```python
def function(arg_1, arg_2 = 2)
"""Count the number of times `letter` appears in `content`.

  
Args:
  arg_1 (str): Description of arg_1 that can break onto the next line if needed.
  letter (str): The letter to search for.

Returns:
  bool: optional description of the return valus
  Extra lines are not indented

Raises:
  ValueError: If `letter` is not a one-character string.

Notes:
  see https://www.abc.com for more info

"""

```



### Numpydoc

```python

```

### Retrieving docstrings

#### using `.doc
```python
# Get the "count_letter" docstring by using an attribute of the function

docstring = count_letter().__doc__

border = '#' * 28
print('{}\n{}\n{}'.format(border, docstring, border))
```


#### using `inspect` module

```python
# Inspect the count_letter() function to get its docstring
docstring = inspect.getdoc(count_letter)
```