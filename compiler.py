# Classes for HTML token types

class Token:
  def __init__(self, start_pos = None, end_pos = None, line = None, col = None):
    self.start_pos = start_pos
    self.end_pos = end_pos
    self.line = line
    self.col = col
  def __repr__(self):
      parts = [f"type={self.type}"]
      if hasattr(self, "name"):
          parts.append(f"name={self.name}")
      if hasattr(self, "attrs") and self.attrs:
          parts.append(f"attrs={self.attrs}")
      if hasattr(self, "data"):
          parts.append(f"data={self.data}")
      if hasattr(self, "self_closing") and self.self_closing == True:
          parts.append(f"self_closing={self.self_closing}")
      return "<" + " ".join(parts) + ">"

class DOCTYPEToken(Token):
  def __init__(self, name='html', **kwargs):
    super().__init__(**kwargs)
    self.type = "DOCTYPE"
    self.name = name

class StartTagToken(Token):
  def __init__(self, name, attrs=None, self_closing=False, **kwargs):
    super().__init__(**kwargs)
    self.type = "start_tag"
    self.name = name
    self.attrs = attrs or {}
    self.self_closing = self_closing

class EndTagToken(Token):
  def __init__(self, name, **kwargs):
    super().__init__(**kwargs)
    self.type = "end_tag"
    self.name = name

class CommentToken(Token):
  def __init__(self, data, **kwargs):
    super().__init__(**kwargs)
    self.type = "comment"
    self.data = data

class CharacterToken(Token):
  def __init__(self, data, **kwargs):
    super().__init__(**kwargs)
    self.type = "character"
    self.data = data

class EOFToken(Token):
  # Tells the program that its the end of the file
  pass

# Core of the HTML Tokenizer

class HTMLTokenizer():
  def __init__(self, code):
    self.code = code
    self.curr_pos = 0
    self.line = 1
    self.col = 1
    self.state = self.data_state
    self.curr_token = None
    self.curr_attr_name = None
    self.curr_attr_val = None

  def consume(self):
    if self.curr_pos < len(self.code):
      char = self.code[self.curr_pos]
      self.curr_pos += 1
      if char == "\n":
        self.line += 1
        # resets col to 1 when entering a new line
        self.col = 1
      else:
        self.col += 1
      return char
    return None

  def reverse(self):
    if self.curr_pos > 0:
      self.curr_pos -= 1

  def tokenize(self):
    while self.curr_pos < len(self.code):
        result = self.state()
        if result is None:
            result = []
        for token in result:
            yield token

  # HTML States

  def data_state(self):
    char = self.consume()
    if char == "<":
      self.state = self.tag_open_state
      return []
    elif char is None:
      return []
    elif char in [" ", "\n", "\t", "\r"]:
        return []
    else:
      return [CharacterToken(char, start_pos=self.curr_pos - 1, end_pos=self.curr_pos, line=self.line, col=self.col)]

  def tag_open_state(self):
    start = self.curr_pos - 1
    char = self.consume()
    if char == "/":
        self.state = self.end_tag_open_state
        return []
    elif char == "!":
        # Check for DOCTYPE or comment
        if self.code[self.curr_pos:self.curr_pos + 7].upper() == "DOCTYPE":
            self.curr_pos += 7
            self.state = self.DOCTYPE_state
            return []
        elif self.code[self.curr_pos:self.curr_pos + 2] == "--":
            self.curr_pos += 2
            self.state = self.comment_state
            return []
        else:
            return [CharacterToken("<!" + (char or ""))]
    elif char and char.isalpha():
        self.curr_token = StartTagToken(name="",  start_pos=start,  line=self.line, col=self.col)
        self.reverse()
        self.state = self.tag_name_state
        return []
    else:
        return [CharacterToken("<" + (char or ""))]

  def tag_name_state(self):
      char = self.consume()
      if char is None:
          return []
      if char.isspace():
          self.state = self.before_attribute_name_state
          return []
      elif char == "/":
          self.state = self.self_closing_tag_state
          return []
      elif char == ">":
          token = self.curr_token
          token.end_pos = self.curr_pos
          self.curr_token = None
          self.state = self.data_state
          return [token]
      else:
          self.curr_token.name += char
          return []

  def end_tag_open_state(self):
      char = self.consume()
      if char and char.isalpha():
          self.curr_token = EndTagToken(name="", start_pos=self.curr_pos - 2,  line=self.line, col=self.col)
          self.reverse()
          self.state = self.tag_name_state
          return []
      elif char == ">":
          self.state = self.data_state
          return []
      else:
          return [CharacterToken("</" + (char or ""))]

  ############################
  # AI COPAS BELOW ###########
  ############################

  def before_attribute_name_state(self):
    char = self.consume()
    if char is None:
        return []
    elif char.isspace():
        return []  # Skip spaces between attributes
    elif char == "/":
        self.state = self.self_closing_tag_state
        return []
    elif char == ">":
        # End of tag → emit token
        token = self.curr_token
        self.curr_token = None
        self.state = self.data_state
        return [token]
    else:
        # Start new attribute
        self.curr_attr_name = char
        self.curr_attr_value = ""
        self.state = self.attribute_name_state
        return []

  def attribute_name_state(self):
      char = self.consume()
      if char is None:
          return []
      if char.isspace() or char in ["=", ">", "/"]:
          # End of name
          self.reverse()
          self.state = self.after_attribute_name_state
          return []
      else:
          self.curr_attr_name += char
          return []

  def after_attribute_name_state(self):
      char = self.consume()
      if char is None:
          return []
      if char.isspace():
          return []
      elif char == "=":
          self.state = self.before_attribute_value_state
          return []
      elif char == ">":
          self.curr_token.attrs[self.curr_attr_name] = self.curr_attr_value
          token = self.curr_token
          self.curr_token = None
          self.state = self.data_state
          return [token]
      else:
          # New attribute starting
          self.curr_token.attrs[self.curr_attr_name] = self.curr_attr_value
          self.curr_attr_name = char
          self.curr_attr_value = ""
          self.state = self.attribute_name_state
          return []

  def before_attribute_value_state(self):
      char = self.consume()
      if char in ['"', "'"]:
          self.quote_char = char
          self.state = self.attribute_value_quoted_state
          return []
      elif char.isspace():
          return []  # Skip spaces before value
      else:
          self.curr_attr_value = char
          self.state = self.attribute_value_unquoted_state
          return []

  def attribute_value_quoted_state(self):
      char = self.consume()
      if char == self.quote_char:
          self.curr_token.attrs[self.curr_attr_name] = self.curr_attr_value
          self.state = self.before_attribute_name_state
          return []
      elif char is None:
          return []
      else:
          self.curr_attr_value += char
          return []

  def attribute_value_unquoted_state(self):
      char = self.consume()
      if char.isspace():
          self.curr_token.attrs[self.curr_attr_name] = self.curr_attr_value
          self.state = self.before_attribute_name_state
          return []
      elif char == ">":
          self.curr_token.attrs[self.curr_attr_name] = self.curr_attr_value
          token = self.curr_token
          self.curr_token = None
          self.state = self.data_state
          return [token]
      else:
          self.curr_attr_value += char
          return []

  def self_closing_tag_state(self):
    char = self.consume()
    if char == ">":
        self.curr_token.self_closing = True
        self.curr_token.end_pos = self.curr_pos
        self.state = self.data_state
        return [self.curr_token]
    else:
        self.state = self.before_attribute_name_state
        self.reverse()
        return []

  def comment_state(self):
    start = self.curr_pos - 4  # account for "<!--"
    data = ""
    while True:
        char = self.consume()
        if char is None:
            break
        if self.code[self.curr_pos-3:self.curr_pos] == "-->":
            break
        data += char
    end = self.curr_pos
    self.state = self.data_state
    return [CommentToken(data.strip("-"),  start_pos=start, end_pos=end, line=self.line, col=self.col)]

  def DOCTYPE_state(self):
      # Capture where the DOCTYPE starts (before consuming)
      start_line = self.line
      start_col = self.col
      start = self.curr_pos - 9  # approximate since we already saw "<!DOCTYPE"

      # Consume until '>'
      while True:
          char = self.consume()
          if char == ">" or char is None:
              break

      end = self.curr_pos
      end_line = self.line
      end_col = self.col

      # Create the token using the *starting* position
      token = DOCTYPEToken(
          name="html",
          start_pos=start,
          end_pos=end,
          line=start_line,   # start of the token
          col=start_col      # start of the token
      )

      self.state = self.data_state
      return [token]

# Parser
class ParseError(Exception):
    """Raised when the HTML parser encounters a syntax or grammar error."""
    def __init__(self, message, token=None):
        if token is not None:
            pos_info = []
            if getattr(token, "line", None) is not None:
                pos_info.append(f"line {token.line}")
            if getattr(token, "col", None) is not None:
                pos_info.append(f"col {token.col}")
            if pos_info:
                message += f" (at {' '.join(pos_info)})"
            message += f" → token: {token}"
        super().__init__(message)
        self.token = token




class HTMLParser:
    """
    Implements the RDP algorithm based on our EBNF grammar.

    Grammar Rules Implemented:
    - document ::= DOCTYPE node*
    - node     ::= element | COMMENT | TEXT
    - element  ::= ( START_TAG nodes END_TAG ) | SELF_CLOSING_TAG
    - nodes    ::= node*
    """
    def __init__(self, tokens):
        self.tokens = (t for t in tokens)
        self.current_token = next(self.tokens)

    def consume(self, expected_type=None):
      # Consumes current token and advances to the next.
      # Check if the token consumed matches the expected type.
      token = self.current_token

      if expected_type and not isinstance(token, expected_type):
        raise ParseError(
            f"Expected {expected_type.__name__} but found {token.__class__.__name__}",
            token=token
        )
      try:
        self.current_token = next(self.tokens)
      except StopIteration:
        # If iteration stops, it means there are no more tokens available
        self.current_token = EOFToken()

      return token

    def peek(self):
      # Look at current token without consuming it
      return self.current_token

    def parse(self):
      # Implements: document ::= DOCTYPE node*

      # DOCTYPE is required in our grammar
      self.consume(DOCTYPEToken)
      # Parse 'node*'
      self.parse_nodes(stop_at_types=[EOFToken])

      self.consume(EOFToken)

      return

    def parse_nodes(self, stop_at_types):
      # Implements: nodes    ::= node*
        while not isinstance(self.peek(), tuple(stop_at_types)):
          self.parse_node()

        return

    def parse_node(self):
      # Implements: node ::= element | COMMENT | TEXT
      token = self.peek()

      if token.type == "start_tag":
        # Choice element
        self.parse_element()
      elif token.type == "comment":
        # Choice comment
        self.consume()
      elif token.type == "character":
        # Choice text
        self.consume()
      else:
        raise ParseError(
            f"Unexpected token in 'node' context: {token}",
            token=token
        )

    def parse_element(self):
      # Implements: element  ::= ( START_TAG nodes END_TAG ) | SELF_CLOSING_TAG

      token = self.peek()

      if token.type == "start_tag" and token.self_closing == True:
        # Checks for self closing tag
        self.consume()
        return
      # Consume start tag
      start_token = self.consume()
      # Parse nodes
      self.parse_nodes(stop_at_types=[EndTagToken, EOFToken])
      # Consume end tag
      end_token = self.consume(EndTagToken)

      # 4. Check for tag mismatch
      if end_token.name != start_token.name:
        raise ParseError(
            f"Mismatched tag. Expected </{start_token.name}> but found </{end_token.name}>",
            token=end_token
        )

      return



from textwrap import dedent
import sys
import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext


def run_parser(name, html_code):
    """
    Helper function to run the lexer and parser on a test case.
    """
    print("="*40)
    print(f"RUNNING TEST: {name}")
    print("="*40)

    html_code = dedent(html_code)
    print("--- Input HTML ---")
    print(html_code)

    try:
        # 1. Lexer: Create tokens
        print("\n--- Tokens ---")
        tokenizer = HTMLTokenizer(html_code)
        tokens = list(tokenizer.tokenize())

        for t in tokens:
            print(t)

        # 2. Parser: Validate the token stream
        print("\n--- Parse Result ---")
        parser = HTMLParser(tokens)

        parser.parse()

        print("\n[SUCCESS] Parse complete. HTML is valid.\n")

    except ParseError as e:
        print(f"\n[ERROR] Parse FAILED: {e}\n", file=sys.stderr)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}\n", file=sys.stderr)

# --- Test Case 1: Valid HTML ---
valid_html = """
<!DOCTYPE html>
    <html>
        <head>
            <!-- This is a comment -->
            <title>My Page</title>
            <link rel="stylesheet" href="style.css" />
        </head>
        <body>
            <h1 id="main-title">Hello World!</h1>
            <p>This is a paragraph with a <a href="/more">link</a>.</p>
            <br />
        </body>
    </html>
"""

# --- Test Case 2: Malformed HTML (Mismatched Tag) ---
malformed_html = """
    <!DOCTYPE html>
    <html>
        <body>
            <h1>This is a title</h1>
            <p>This is a paragraph.</p>
        </body>
    </div>
"""

# --- Test Case 3: Malformed HTML (Missing DOCTYPE) ---
missing_doctype_html = """
<html>
        <head><title>Test</title></head>
        <body><p>Hello</p></body>
    </html>
"""

if __name__ == "__main__":
    if "--tests" in sys.argv:
        run_parser("Valid HTML", valid_html)
        run_parser("Malformed HTML (Mismatched Tag)", malformed_html)
        run_parser("Malformed HTML (Missing DOCTYPE)", missing_doctype_html)
    else:
        class HTMLParserGUI:
            def __init__(self):
                self.root = tk.Tk()
                self.root.title("HTML Parser IDE")
                self.root.geometry("1200x700")
                self.root.configure(bg="#1e1e1e")

                self.base_font = ("Consolas", 11)

                self._build_layout()

            def _build_layout(self):
                toolbar = ttk.Frame(self.root, padding=10)
                toolbar.pack(fill=tk.X)

                run_button = ttk.Button(toolbar, text="Run Parser", command=self.run_parser)
                run_button.pack(side=tk.LEFT)

                clear_button = ttk.Button(toolbar, text="Clear Outputs", command=self.clear_outputs)
                clear_button.pack(side=tk.LEFT, padx=(10, 0))

                main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
                main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # HTML editor
                editor_frame = ttk.Frame(main_pane)
                self._add_label(editor_frame, "HTML Source")
                self.html_text = self._create_scrolled_text(editor_frame)
                self.html_text.pack(fill=tk.BOTH, expand=True)

                main_pane.add(editor_frame, weight=2)

                # Side pane for tokens and errors
                side_pane = ttk.Panedwindow(main_pane, orient=tk.VERTICAL)

                tokens_frame = ttk.Frame(side_pane)
                self._add_label(tokens_frame, "Tokens")
                self.tokens_text = self._create_scrolled_text(tokens_frame)
                self.tokens_text.pack(fill=tk.BOTH, expand=True)
                self.tokens_text.config(state=tk.DISABLED)
                side_pane.add(tokens_frame, weight=3)

                errors_frame = ttk.Frame(side_pane)
                self._add_label(errors_frame, "Parser Output")
                self.error_text = self._create_scrolled_text(errors_frame, height=8)
                self.error_text.pack(fill=tk.BOTH, expand=True)
                self.error_text.config(state=tk.DISABLED)
                side_pane.add(errors_frame, weight=1)

                main_pane.add(side_pane, weight=1)

                default_html = dedent("""\
                <!DOCTYPE html>
                <html>
                    <head>
                        <title>Sample</title>
                    </head>
                    <body>
                        <h1>Hello, world!</h1>
                    </body>
                </html>
                """)
                self.html_text.insert(tk.END, default_html)

            def _add_label(self, parent, text):
                lbl = ttk.Label(parent, text=text, font=("Segoe UI", 10, "bold"))
                lbl.pack(anchor="w", padx=5, pady=(5, 0))

            def _create_scrolled_text(self, parent, height=None):
                widget = scrolledtext.ScrolledText(
                    parent,
                    wrap=tk.NONE,
                    height=height,
                    font=self.base_font,
                    background="#1e1e1e",
                    foreground="#d4d4d4",
                    insertbackground="#d4d4d4",
                    borderwidth=1,
                    relief=tk.FLAT,
                )
                return widget

            def clear_outputs(self):
                for widget in (self.tokens_text, self.error_text):
                    widget.config(state=tk.NORMAL)
                    widget.delete("1.0", tk.END)
                    widget.config(state=tk.DISABLED)

            def run_parser(self):
                html_input = self.html_text.get("1.0", tk.END).strip()
                self.clear_outputs()

                if not html_input:
                    self._append_error("No HTML input provided.")
                    return

                try:
                    tokenizer = HTMLTokenizer(html_input)
                    tokens = list(tokenizer.tokenize())
                    self._append_tokens(tokens)

                    parser = HTMLParser(tokens)
                    parser.parse()
                    self._append_error("[SUCCESS] Parse complete. HTML is valid.")
                except ParseError as err:
                    self._append_error(f"[ERROR] {err}")
                except Exception as err:
                    self._append_error(f"[CRITICAL ERROR] {err}")

            def _append_tokens(self, tokens):
                self.tokens_text.config(state=tk.NORMAL)
                for token in tokens:
                    self.tokens_text.insert(tk.END, f"{token}\n")
                self.tokens_text.config(state=tk.DISABLED)

            def _append_error(self, message):
                self.error_text.config(state=tk.NORMAL)
                self.error_text.insert(tk.END, message + "\n")
                self.error_text.config(state=tk.DISABLED)

            def start(self):
                self.root.mainloop()

        gui = HTMLParserGUI()
        gui.start()