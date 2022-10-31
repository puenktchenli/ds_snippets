
## Writing Python functions

#clean_code_function #return_annotation

```python
#return annotation
def kinetic_energy(m:'in KG', v:'in M/S')->'Joules': 
    return 1/2*m*v**2
 
>>> kinetic_energy.__annotations__
{'return': 'Joules', 'v': 'in M/S', 'm': 'in KG'}
```


## Underscore( __ ) in Python
#underscore

[readMore](https://www.datacamp.com/tutorial/role-underscore-python)

### Use in interpreter
**Python** automatically stores the value of the last expression in the **interpreter** to a particular variable called "_."  You can also assign these value to another variable if you want.

```python
>>> 5 + 4
9
>>> _     # stores the result of the above expression
9
>>> _ + 6
15
>>> _
15
>>> a = _  # assigning the value of _ to another variable
>>> a
15
```


Now, if you **import** all the methods and names from **my_functions.py**, **Python** doesn't import the names which starts with a **single pre underscore**.