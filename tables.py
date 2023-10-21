from .utils import string_to_python_value
import warnings


class Matrix:

  @staticmethod
  def matrix_or_error(vectors_stack):
    if not isinstance(vectors_stack, list):
      raise TypeError(f'vectors_stack should be a list of lists, but got {type(vectors_stack)}')

    last = len(vectors_stack[0])
    for vector in vectors_stack[1:]:
      if last != len(vector):
        raise ValueError('vectors should be the same length')
      last = len(vector)
    return vectors_stack


  def __init__(self, vectors_stack: list):
    """
    :params: vectors_stack — list lists like [[1, 2], [3, 4]] where each list is a row, and elements are columns
    """
    self.matrix = Matrix.matrix_or_error(vectors_stack)

  @property
  def shape(self):
    """updates shape if self.matrix was changed"""
    return (len(self.matrix), len(self.matrix[0]))

  @property
  def transposed(self):
    t = []
    for col in range(self.shape[1]):
      cols_data = []
      for row in self.matrix:
        cols_data.append(row[col])
      t.append(cols_data)
    return t

  def __str__(self):
    return '\n'.join([str(vector) for vector in self.matrix])

  def __repr__(self):
    return str(self)

  def __getitem__(self, index: tuple) -> Any:
    """returns elements based on column and row"""
    if isinstance(index, int):
      raise TypeError('matrix cannot understand that slicing, you should specify column(s) and row(s)')

    col, row = index
    if isinstance(row, slice):
      t = self.transposed[col] if isinstance(col, slice) else [self.transposed[col]]
      t_reduced_rows = []
      for column in t:
        t_reduced_rows.append(column[row])
      return Matrix(t_reduced_rows).transposed # транспонируем обратно
    return self.matrix[row][col]


  def __setitem__(self, index: tuple, value: Any):
    col, row = index
    if isinstance(row, slice):
      # recursion is used
      start = 0 if row.start is None else row.start
      stop = self.shape[0] if row.stop is None else row.stop
      for i_row in range(start, stop):
        if (row.step is None) or (i_row % row.step == 0):
          self[col, i_row] = value

    if isinstance(col, slice) and not isinstance(row, slice):
      start = 0 if col.start is None else col.start
      stop = self.shape[1] if col.stop is None else col.stop
      for i_col in range(start, stop):
        if (col.step is None) or (i_col % col.step == 0):
          self.matrix[row][i_col] = value

    if not (isinstance(col, slice) or isinstance(row, slice)):
      self.matrix[row][col] = value


  def apply(self, func: Callable, axis:str, *args, **kwargs):
    """applies func for each row/column if condition is met
    :params: func — func that would be applied on each row/column (func must have list as first param and cannot return list, dict or tuple)
    :params: axis — if 'row' yield each row for func, if 'col' yield each column for func
    :params: *args, **kwargs — custom func params

    :returns: new matrix"""
    assert axis in ['row', 'col'], 'axis is either row or col'
    assert type(func([1], *args, **kwargs)) in [int, float, str, None], 'func must have list as first param and returns int, float, str or None'

    matrix = self.matrix if axis == 'row' else self.transposed
    new_matrix = []
    for arr in matrix:
      new_matrix.append(
          func(arr, *args, **kwargs)
          )
    return new_matrix


  def insert(self, axis: str, where: tuple, dump = None, *args, **kwargs):
    """inserts col or row wherever you want, dumps it with default value = dump
    :params: axis — row/columns
    :params: where — index where to insert axis
    :params: dump -- dump value for new row/column, if function, applies it to the whole axis to get value
    """
    assert axis in ['row', 'col'], 'axis is either row or col'
    assert (type(dump) in [int, float, str, None]) or isinstance(dump, Callable), 'dump must be int, str, float, None or function'

    length = self.shape[0] if axis == 'row' else self.shape[1]
    index = where if where >= 0 else (length + 2 if where == -1 else where + 1) # 0 - start, 1 - second, but -1 - after last, -2 pre_last
    new_arr = [dump]*length
    if axis == 'row':
      to_insert = [dump]*self.shape[1] if not isinstance(dump, Callable) else self.apply(dump, 'col', *args, **kwargs)
      self.matrix.insert(
          index,
          to_insert
      )

    else:
      t = self.transposed
      to_insert = [dump]*len(t[0]) if not isinstance(dump, Callable) else Matrix(t).apply(dump, 'col', *args, **kwargs)
      t.insert(
          index,
          to_insert
      )
      self.matrix = Matrix(t).transposed

    return self

  def copy(self):
    return Matrix(self.matrix.copy())


"""----------------------------------------------------------------------------"""


class HandyTable(Matrix):

  """===========================================================================
  defining class attribute
  """

  ALPHABET = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") # all possibles columns

  #-------------------------------css code--------------------------------------

  # css code for table and table cells
  css = """<style>
  .TableCell:hover {
    background-color: #D6EEEE;
    }
  table, th, td {
    padding: 10px;
    border: 1px solid black;
    border-collapse: collapse;
    }
  </style>
  """

  # ------------------------------- js code-----------------------------------
  if 'google.colab' in sys.modules:
    from google.colab import output
    INVOKER = 'google.colab.kernel.invokeFunction'
    REGISTRATE = output.register_callback

    js = """
    <script>
    document.querySelectorAll("td[contenteditable].forEach(function(element){{
      element.addEventListener("input", function(e){{
        const val = e.target.innerText;
        const id = e.target.id;
        {invoker}("setVal", [val, id], {{}});
      }});
    }});
    );
    </script>
    """
  else:
    warnings.warn("HandyTable js-feature is available only in google.colab for now")
    js = ""
     

  #------------------------------class util-------------------------------------



  def string_to_python_value(val: str):
    """converts string to appropriate value, it is used for js-python communication"""
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


  """===========================================================================
  class static methods
  """

  @staticmethod
  def as_table(columns, indexes, matrix):
    """shows matrix with given columns and indexes"""
    if len(columns) > len(matrix[0]):
      raise IndexError('no such columns')
    if len(indexes) > len(matrix):
      raise IndexError('no such rows')

    columns = [''] + columns
    table = str(columns) + '\n'
    for n, row in zip(indexes, matrix):
      row = [n] + row
      table += str(row) + '\n'
    return table


  @staticmethod
  def as_html_table(columns, indexes, matrix):
    base_html = """
    <table>
    {columns}
    {rows}
    </table>
    """
    columns_html = '<th>{col}</th>'
    index_html = '<td><strong>{index}</strong></td>'
    if bool(HandyTable.js):
      cells_html = """<td class="TableCell", id="{id}" contenteditable>{cell_data}</td>"""
    else:
      cells_html = """<td class="TableCell", id="{id}">{cell_data}</td>"""

    cols = '<tr>' + '\n'.join([columns_html.format(col=col) for col in ['']+columns]) + '</tr>'

    rows = []
    for n, row in zip(indexes, matrix):
      index = index_html.format(index=n)
      data = '\n'.join([cells_html.format(id=f'{columns[c]}:{n}', cell_data=data) for c, data in enumerate(row)])
      rows.append('<tr>'+index+data+'</tr>')
    rows = '\n'.join(rows)

    return base_html.format(columns=cols, rows=rows)


  """===========================================================================
  Inner Class, it's used for manipulation with piece of HandyTable
  """

  class SubTable():

    def __init__(self, start, end, parent):
      self.parent = parent # HandyTable
      # selection
      self.start = start
      self.end = end

    # ---------------------------innner class properties------------------------

    @property
    def slicings(self):
      start_col = HandyTable.ALPHABET.index(self.start[0])
      start_row = int(self.start[1])
      end_col = HandyTable.ALPHABET.index(self.end[0])+1
      end_row = int(self.end[1])+1
      return start_col, end_col, start_row, end_row

    @property
    def matrix(self):
      sc, ec, sr, er = self.slicings
      return self.parent[sc:ec, sr:er]

    @property
    def columns(self):
      return HandyTable.ALPHABET[HandyTable.ALPHABET.index(self.start[0]):HandyTable.ALPHABET.index(self.end[0])+1]

    @property
    def indexes(self):
      return range(int(self.start[1]), int(self.end[1])+1)

    # ---------------------------inner class apperance--------------------------

    def __str__(self):
      return HandyTable.as_table(self.columns, self.indexes, self.matrix)

    def _repr_html_(self):
      def set_val_sub(id:str, val:Any):
        """takes argument from js code, assign given value to self"""
        col, row = id.split(':') # id is taken from generated html
        col = HandyTable.ALPHABET.index(col)
        row = int(row)
        # val = HandyTable.string_to_python_value(val)
        val = string_to_python_value(val)
        self.parent[col, row] = val # assigning recieved value to parent matix

      if bool(HandyTable.js):
        HandyTable.REGISTRATE('selVal', set_val_sub)

      html = HandyTable.as_html_table(self.columns, self.indexes, self.matrix)
      return HandyTable.css + html + HandyTable.js

   #-------------------------------utils----------------------------------------

    def assign(self, value) -> None: # python does not let me make __setitem__ ):
      """assigns value for the selected range in parent table"""
      # assert type(value) in [int, str, float, None], 'value should be int, str, float, None or Callable'
      sc, ec, sr, er = self.slicings
      self.parent[sc:ec, sr:er] = value
      return None

    def apply(self, func: Callable, *args, **kwargs):
      """flattens sliced matrix and applied func to the result list"""
      tested_func_res = func([1], *args, **kwargs)
      assert type(tested_func_res) in [int, float, str, None], f'func must take a list and return str, int, float or None, but returned {tested_func_res}'
      flattened = []
      for row in self.matrix:
        for e in row:
          flattened.append(e)
      return func(flattened,  *args, **kwargs)

  """===========================================================================
  HandyTable instance
  """

  def __init__(self, matrix_values: list):
    super().__init__(matrix_values)


  #-------------------------------properties-----------------------------------
  @property
  def columns(self):
    return self.ALPHABET[:self.shape[1]]

  @property
  def indexes(self):
    return range(self.shape[0])

  #-------------------------------apperance------------------------------------
  def __str__(self):
    return HandyTable.as_table(self.columns, self.indexes, self.matrix)

  def _repr_html_(self):
    def set_val(val:Any, id:str):
      """takes argument from js code, assign given value to self"""
      col, row = id.split(':') # id is taken from generated html
      col = HandyTable.ALPHABET.index(col)
      row = int(row)
      # val = HandyTable.string_to_python_value(val)
      val = string_to_python_value(val)
      self[col, row] = val

    if bool(HandyTable.js):
      HandyTable.REGISTRATE('setVal', set_val)

    html = HandyTable.as_html_table(self.columns, self.indexes, self.matrix)
    return self.css + html + self.js

  #------------------------------specific method--------------------------------

  def select(self, selection):
    """evaluates self range
       example of correct syntax: A1-B2, A2, B6-C6, A0
       wrong syntax: A1-, B2-A1, a1-G2, ф1, A-1
       !!! cyrrilic sensitive
       """
    pat = re.compile("""
    ([A-Z]\d+)              # range group 1 e.g. A1
    -?
    ([A-Z]\d+)?             # not necessary 2 range group e.g -B2
    """, re.VERBOSE)
    matched = pat.match(selection)

    if matched is None:
      raise ValueError(f'{selection} is not correct range, use upper-case letters A-Z and following row number e.g. A1, B2 or A1-B2')

    if (matched.group(2) is not None) and ('-' not in selection):
      raise ValueError(f'''{selection} should countain '-' between start and end cells''')
    if (matched.group(2) is None) and ('-' in selection):
      raise ValueError(f'{selection} has incorrect endpoint, check for cyrrilic!')

    start = matched.group(1)
    end = start if matched.group(2) is None else matched.group(2)

    return HandyTable.SubTable(start, end, self)
