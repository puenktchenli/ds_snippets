---
title: Clean and Efficient Codes
tags: [formatting]
keywords: clean_code, tips, PEP
last_updated: Oct 31, 2022
summary: "A brief guide for clean codes"
sidebar: mydoc_sidebar
permalink: cs_clean_code.html
---

#clean #efficient #pythonic #zen

# Efficient code

> [!Note]- The Zen of Python
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!

## Built-ins
### Built-in types
`list`, `tuple`, `set`, `dict` , and others
### Built-in funcitons
- `range()

- `enumerate()

```python
indexed_letters2 = enumerate(letters, start=5)
```

- `map`

- `map` with `lambda`

```python
sqrd_nums = map(lambda x: x ** 2, nums)
```

### Built-in modules
- `os` , `sys`, `itertools` , `collections`, `math` and others



## packing
### Packing with * Operator ([more...](https://stackabuse.com/unpacking-in-python-beyond-parallel-assignment/))
```python
*a, b = 1, 2, 3 

>>> a 
[1, 2] 
>>> b
 3
```

```python
# to unpack the object of enumerate to a list, using * 
indexed_names_unpack = [*enumerate(names, start=1)]
```


## Examining runtime  `%timeit 
([more](https://campus.datacamp.com/courses/writing-efficient-python-code/timing-and-profiling-code?ex=2))
- Caluculate runtime with IPython magic command `%timeit
- **Magic commands**: enhancements on top of normal Python syntax
  - Prefixed by the `%` character
  - Link to docs
  - See all available magic commands with `%lsmagic`


```python
# run time estimated the time 
%timeit rand_nums = np.random.rand(1000)

```

### Specifying number of runs/loops
- Setting the number of runs(`-r`) and /or loops (`-n`)
```python
%timeit -r2 -n10 rand_nums = np.random.rand(1000)

```

### Saving  `%timeit` output by using `-o
```python
times = %timeit -o rand_nums = np.random.rand(1000)
times.timings
times.best
times.worst

```

### Using cell magic mode `%%timeit`
press `SHIFT+ENTER` l

```python
%%timeit
hero_wts_lbs = []
for wt in wts:
    hero_wts_lbs.append(wt * 2.20462)
```



## 