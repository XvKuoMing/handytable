def string_to_python_value(val: str):
  """coverts string to appropriate value, it is used for js-python communication"""
  # -----
  # future code for prohibiting html-injection
  # -----
  if val.replace('.', '').isnumeric(): # only data such as '12', '12.1', '123.', '12.5634' is allowed
    try:
      val = int(val)
    except ValueError:
      val = float(val)
  else:
      val = val if bool(val) else None
  return val


"""Implementing Excel-like formula's syntax
-------------------------------------------------------------------------------"""

avg = lambda arr: sum(arr)/len(arr) # takes a list of values and returns avg from them
def product(arr):
  """multiplies each element in the list"""
  res = arr[0]
  for e in arr[1:]:
    res *= e
  return res

FORMULA_MAPPER = {'SUM': sum, 'COUNT': len, 'AVG': avg, 'PRODUCT': product}
