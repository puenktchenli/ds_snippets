
---
dg-home: true
dg-publish: true
---

#context_manager #with

Context manager:
- Sets up a context
- Runs your code 
- Removes the context


## Using Context Managers

```python
# use as to assign a return value, which is called open context
with <context-manager>(<args>) as <variable-name> 

    # run your code here
    # This code is running "inside the context"

# This code runs after the context is removed
```

Example 1: 

```python
# Open "alice.txt" and assign the file to "file"
with open('alice.txt') as file:
    text = file.read()

n = 0
for word in text.split():
     if word.lower() in ['cat', 'cats']:
         n += 1

print('Lewis Carroll uses the word "cat" {} times'.format(n))
```

```python
image = get_image_from_instagram()

# Because timer() is a context manager that does not return a value.

# Time how long process_with_numpy(image) takes to run
with timer():
    print('Numpy version')
    process_with_numpy(image)

# Time how long process_with_pytorch(image) takes to run
with timer():
    print('Pytorch version')
    process_with_pytorch(image)
```


## Writing Context Managers
	for other people to use

### when to use: 

|Patterns         |          |
| ------- | -------- |
| Open    | Close    |
| Lock    | Release  |
| Change  | Reset    |
| Enter   | Exit     |
| Start   | Stop     |
| Setup   | Teardown |
| Connect | Disconnect         |


two ways to define a context manager
- Class-based2
- Function - based

### function - based context managers

```python
@contextlib.contextmanager  # Add the decorator
def my_context():
    # Add any set up code
    yield # yield one value or just yield
    # Add any teardown code to clean up the context

```

### Nested Contexts
nested `with` 

```python
defcopy(src, dst):
"""Copy the contents of one file to another. 

Args: 
    src (str): File name of the file to be copied. 
    dst (str): Where to write the new file. 
    """
    # Open the source file and read in the contents
    with open(src) as f_src: 
        contents = f_src.read()
        
        # Open the destination file and write out the contents
        with open(dst, 'w') as f_dst: 
            f_dst.write(contents)
```


### Handling errors
#try #except #finally

```python
try:
    # code that might raise an error
except: 
    # do something about the error
finally:
    # this code runs no matter what
```

Put `try` statement in front of yield statement, and put `finally` statement before teardown code. By doing this, we can ensure the teardown with always happen when even errors occurred. 


