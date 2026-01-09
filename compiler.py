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
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.type = "EOF"

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
        raise ParseError("Unexpected EOF inside tag", self.curr_token)

    # illegal: newline inside tag name
    if char == "\n":
        raise ParseError(
            f"Malformed start tag <{self.curr_token.name}> (missing '>')",
            self.curr_token
        )

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

    elif char == "<":
        raise ParseError(
            f"Malformed start tag <{self.curr_token.name}> (unexpected '<')",
            self.curr_token
        )

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

  #

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
    elif char == "\n":
        raise ParseError(
            f"Malformed start tag <{self.curr_token.name}> (missing '>')",
            self.curr_token
        )

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
        # Check for duplicate
          if self.curr_attr_name in self.curr_token.attrs:
             raise ParseError(f"Duplicate attribute '{self.curr_attr_name}'", self.curr_token)

          self.curr_token.attrs[self.curr_attr_name] = self.curr_attr_value
          token = self.curr_token
          self.curr_token = None
          self.state = self.data_state
          return [token]
      else:
          # New attribute starting
          # Check for duplicate
          if self.curr_attr_name in self.curr_token.attrs:
              raise ParseError(f"Duplicate attribute '{self.curr_attr_name}'", self.curr_token)

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
          # CHECK FOR DUPLICATE BEFORE ASSIGNING
          if self.curr_attr_name in self.curr_token.attrs:
            raise ParseError(f"Duplicate attribute '{self.curr_attr_name}'", self.curr_token)

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
          # CHECK FOR DUPLICATE
          if self.curr_attr_name in self.curr_token.attrs:
            raise ParseError(f"Duplicate attribute '{self.curr_attr_name}'", self.curr_token)

          self.curr_token.attrs[self.curr_attr_name] = self.curr_attr_value
          self.state = self.before_attribute_name_state
          return []
      elif char == ">":
          # CHECK FOR DUPLICATE
          if self.curr_attr_name in self.curr_token.attrs:
            raise ParseError(f"Duplicate attribute '{self.curr_attr_name}'", self.curr_token)

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
    start_line = self.line
    start_col = self.col
    start = self.curr_pos - 9

    while True:
        char = self.consume()

        if char == ">":
            break

        if char is None or char == "<" or self.line != start_line:
            raise ParseError(
                "Malformed DOCTYPE declaration (missing '>')",
                token=DOCTYPEToken(
                    start_pos=start,
                    line=start_line,
                    col=start_col
                )
            )

    end = self.curr_pos

    token = DOCTYPEToken(
        name="html",
        start_pos=start,
        end_pos=end,
        line=start_line,
        col=start_col
    )

    self.state = self.data_state
    return [token]



# Constants for semantic analysis

# Void elements - tags that dont need a closing tag
VOID_ELEMENTS = {
    "area","base","br","col","embed","hr","img",
    "input","link","meta","source","track","wbr"
}

# Block elements - block elements can contain inline, but incline cant contain block
BLOCK_ELEMENTS = {
    "address","article","aside","blockquote","canvas","dd","div","dl",
    "dt","fieldset","figcaption","figure","footer","form","h1","h2",
    "h3","h4","h5","h6","header","hr","li","main","nav","noscript",
    "ol","p","pre","section","table","tfoot","ul","video"
}

# Strict parent rules - child must be inside one of these
# key -> element, value -> parents
REQUIRED_PARENTS = {
    "li" : {"ul", "ol"},
    "dt" : {"dl"},
    "dd" : {"dl"},
    "tr" : {"table", "tbody", "thead", "tfoot"},
    "td" : {"tr"},
    "th" : {"tr"},
    "title" : {"head"},
    "meta"  : {"head"}
}

ALLOWED_CHILDREN = {
    "ul":    {"li"},
    "ol":    {"li"},
    "table": {"tr", "tbody", "thead", "tfoot"},
    "tbody": {"tr"},
    "thead": {"tr"},
    "tfoot": {"tr"},
    "tr":    {"td", "th"}
}

# A simple schema defining allowed attributes
HTML_SCHEMA = {
    # Attributes allowed on ANY tag
    "global": {"id", "class", "style", "title", "lang", "hidden"},

    # Attributes specific to certain tags
    "tags": {
        "html": set(),
        "head": set(),
        "body": set(),
        "title": set(),
        "span": set(),
        "label": {"for"},
        "button": {"typed","disabled","name","value"},
        "iframe": {"src","width","height","frameborder","allow"},
        "ol": {"type","start","reversed"},
        "table": {"border", "cellpadding", "cellspacing"},
        "tr": set(),
        "th": {"scope", "colspan", "rowspan"},
        "td": {"colspan", "rowspan"},
        "h1": set(),
        "h2": set(),
        "h3": set(),
        "h4": set(),
        "h5": set(),
        "h6": set(),
        "ul": set(),
        "li": set(),
        "a":   {"href", "target", "rel"},
        "img": {"src", "alt", "width", "height"},
        "div": set(),
        "p":   set(),
        "input": {"type", "value", "placeholder", "required", "name"},
        "form": {"action", "method"},
        "meta": {"charset", "name", "content"},
        "link": {"rel", "href", "type"}
    },

    # Required attributes (Strict mode)
    "required": {
        "img": {"src","alt"},
        "a":   {"href"},
        "iframe": {"src"},
        "label": {"for"}
    }
}

# Constants for value validation
VALID_INPUT_TYPES = {
    "text", "password", "email", "number", "checkbox",
    "radio", "submit", "button", "date", "color",
    "range", "hidden", "file", "search", "tel", "url"
}

class Node:
  pass

class Element(Node):
    def __init__(self, tag, attrs, children, line=None, col=None):
      self.type = "element"
      self.tag = tag
      self.attrs = attrs
      self.children = children
      self.line = line
      self.col = col

    def __repr__(self):
      return f"Element({self.tag}, attrs={list(self.attrs.keys())}, children={len(self.children)})"

class TextNode(Node):
    def __init__(self, text):
      self.type = "text"
      self.text = text

    def __repr__(self):
        return f"Text({repr(self.text)})"

class CommentNode(Node):
    def __init__(self, text):
      self.type = "comment"
      self.data = text

    def __repr__(self):
        return f"Comment({repr(self.data)})"


# Parser
class ParseError(Exception):
    """Raised when the HTML parser encounters a syntax or grammar error."""
    def __init__(self, message, token=None):
        self.token = token
        self.line = None
        self.col = None
        self.length = 1  # default highlight length

        if token is not None:
            if getattr(token, "line", None) is not None:
                self.line = token.line
            if getattr(token, "col", None) is not None:
                self.col = token.col

            pos_info = []
            if self.line is not None:
                pos_info.append(f"line {self.line}")
            if self.col is not None:
                pos_info.append(f"col {self.col}")
            if pos_info:
                message += f" (at {' '.join(pos_info)})"

            message += f" → token: {token}"

            # Optional: smarter highlight length
            if hasattr(token, "value"):
                self.length = len(str(token.value))

        super().__init__(message)




class HTMLParser:
    """
    Implements the RDP algorithm based on our EBNF grammar.

    Grammar Rules Implemented:
    - document ::= DOCTYPE nodes EOF
    - node     ::= element | COMMENT | TEXT
    - element  ::= START_TAG nodes END_TAG  | SELF_CLOSING_TAG | VOID_ELEMENT
    - nodes    ::= {node}
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
      # Implements: document ::= DOCTYPE nodes EOF

      # DOCTYPE is required in our grammar
      self.consume(DOCTYPEToken)

      root_nodes = self.parse_nodes(stop_at_types=[EOFToken])

      self.consume(EOFToken)
      # look for html elements
      html_elements = [node for node in root_nodes if isinstance(node, Element) and node.tag == "html"]

      if len(html_elements) == 0:
        raise ParseError("Document is missing an <html> element.")
      elif len(html_elements) > 1:
        second_html = html_elements[1]
        raise ParseError(
        "Document cannot have multiple <html> elements.",
        token=StartTagToken(
            name="html",
            start_pos=None,
            line=second_html.line,
            col=second_html.col
        )
    )
      # return the root element
      return html_elements[0]

    def parse_nodes(self, stop_at_types, parent_tag=None):
      # Implements: nodes ::= {node}
        children = []
        while not isinstance(self.peek(), tuple(stop_at_types)):
          node = self.parse_node(parent_tag=parent_tag)
          if node:
            children.append(node)
        return children

    def parse_node(self, parent_tag=None):
      # Implements: node ::= element | COMMENT | TEXT
      token = self.peek()

      if token.type == "start_tag":
        # Choice element
        return self.parse_element(parent_tag=parent_tag)
      elif token.type == "comment":
        # Choice comment
        self.consume()
        return CommentNode(token.data)
      elif token.type == "character":
        # Choice text
        self.consume()
        return TextNode(token.data)
      else:
        raise ParseError(
            f"Unexpected token in 'node' context: {token}",
            token=token
        )

    def parse_element(self, parent_tag = None):
      # Implements: element  ::= START_TAG nodes END_TAG  | SELF_CLOSING_TAG | VOID_ELEMENT

      token = self.peek()
      # Handle void elements
      if token.name in VOID_ELEMENTS:
        start_token = self.consume() # Eat the tag (example: <br>)
        return Element(start_token.name, start_token.attrs, children=[], line=start_token.line,
                col=start_token.col)

      # Handle self closing tag elements
      if token.type == "start_tag" and token.self_closing == True:
        start_token = self.consume()
        return Element(start_token.name, start_token.attrs, children=[], line=start_token.line,
                col=start_token.col)

      # Normal elements
      start_token = self.consume()
      current_tag = start_token.name

      # Check required parents (e.g., <li> must be in <ul>)
      if current_tag in REQUIRED_PARENTS:
        allowed = REQUIRED_PARENTS[current_tag]
        if parent_tag not in allowed:
          raise ParseError(
              f"<{current_tag}> must be inside {allowed}, but found in <{parent_tag}>",
              token=start_token
          )

      # Check if a Block is inside an Inline element, raises an error if a block element inside an inline
      if parent_tag and (parent_tag not in BLOCK_ELEMENTS) and (current_tag in BLOCK_ELEMENTS):
        # exception: <body>
         if parent_tag != "body" and parent_tag != "html":
          raise ParseError(
              f"Block element <{current_tag}> cannot be inside inline element <{parent_tag}>",
              token=start_token
          )


      # Parse nodes
      children = self.parse_nodes(stop_at_types=[EndTagToken, EOFToken], parent_tag=current_tag)
      # Consume end tag
      end_token = self.consume(EndTagToken)

      # Check for tag mismatch
      if end_token.name != start_token.name:
        raise ParseError(
            f"Mismatched tag. Expected </{start_token.name}> but found </{end_token.name}>",
            token=start_token
        )

      return Element(current_tag, start_token.attrs, children, line=start_token.line,
                col=start_token.col)



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

        if e.line is not None and e.col is not None:
            lines = html_code.splitlines()

            # Guard against out-of-range
            if 1 <= e.line <= len(lines):
                error_line = lines[e.line - 1]

                print(f"  --> line {e.line}, column {e.col}", file=sys.stderr)
                print(f"   |", file=sys.stderr)
                print(f"{e.line:4} | {error_line}", file=sys.stderr)

                # Build caret underline
                caret_padding = " " * (e.col)
                caret = "^" + "~" * max(e.length - 1, 0)

                print(f"   | {caret_padding}{caret}\n", file=sys.stderr)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] An unexpected error occurred: {e}\n", file=sys.stderr)

def validate_semantics(node, context=None):
  if context is None:
    context = {
        "ids": set(), # for checking duplicate id
        "ancestors":set(), # for nesting check
        "errors":[] # list to collect all errors
    }

  if node.type == 'element':
    tag = node.tag
    attrs = node.attrs.keys()

    # Check if its a valid html tag
    if tag not in HTML_SCHEMA["tags"]:
      context["errors"].append({
        "line": node.line,
        "message": f"Unknown or illegal tag: <{tag}>"
      })

    # Ensuring that there is only 1 head and body element in a html element
    if tag == "html":
      # Filter out comments or text
      found_tags = []
      # Check the children elements of the html element
      for child in node.children:
        if child.type == "element":
          found_tags.append(child.tag)
      # Checks if the order of head and body element is correct and that there is only 1 of each
      if found_tags != ["head","body"]:
        context["errors"].append(
            f"[Line {node.line}] <html> structure invalid. Expected <head> followed by <body>, but found: {found_tags}"
        )

    # Check allowed children
    if tag in ALLOWED_CHILDREN:
      valid_children = ALLOWED_CHILDREN[tag]

      for child in node.children:
        if child.type == "element":
          # If its not in valid children then set an error
          if child.tag not in valid_children:
            context["errors"].append(
                f"[Line {child.line}] <{child.tag}> is not allowed directly inside <{tag}>. Allowed: {valid_children}"
            )

    # Checking for input types
    if tag == "input":
      if "type" in node.attrs:
        val = node.attrs["type"]
        if val not in VALID_INPUT_TYPES:
          context["errors"].append({
              "line": node.line,
              "message": (
                  f"Invalid input type '{val}'. "
                  f"Allowed types: {sorted(VALID_INPUT_TYPES)}"
              )
          })


    # Checking for illegal attributes
    allowed = HTML_SCHEMA["global"].copy()
    if tag in HTML_SCHEMA["tags"]:
      allowed.update(HTML_SCHEMA["tags"][tag])

      for attr in attrs:
        if attr not in allowed:
          context["errors"].append({
                "line": node.line,
                "message": (
                    f"Attribute '{attr}' is NOT allowed on tag <{tag}>."
                )
            })

    # Checking for missing required attributes
    if tag in HTML_SCHEMA["required"]:
      required = HTML_SCHEMA["required"][tag]
      if not required.issubset(attrs):
        missing = required - attrs
        context["errors"].append({
            "line": node.line,
            "message": (
                f"Tag <{tag}> is missing required attribute(s): "
                f"{', '.join(sorted(missing))}"
            )
        })


    # Checking for duplicate IDs
    if "id" in node.attrs:
        id_val = node.attrs["id"]
        if id_val in context["ids"]:
            context["errors"].append({
                "line": node.line,
                "message": f"Duplicate ID found: '{id_val}'"
            })
        context["ids"].add(id_val)


    # Checking for nesting rules
    forbidden_rules = {
        "a": {"a", "button"},
        "form": {"form"},
        "button": {"button","a"}
    }
    if tag in forbidden_rules:
      conflict = context["ancestors"].intersection(forbidden_rules[tag])
      if conflict:
        context["errors"].append({
            "line": node.line,
            "message": (
                f"<{tag}> cannot be nested inside <{list(conflict)[0]}>"
            )
        })


    # Update ancestors for children
    new_ancestors = context["ancestors"].copy()
    new_ancestors.add(tag)

    child_context = {
            "ids": context["ids"],
            "errors": context["errors"],
            "ancestors": new_ancestors
    }

    for child in node.children:
            validate_semantics(child, child_context)

  return context["errors"]

if __name__ == "__main__":
        class HTMLParserGUI:
            def __init__(self):
                self.root = tk.Tk()
                self.root.title("HTML Parser IDE")
                self.root.geometry("1200x700")
                self.root.configure(bg="#1e1e1e")

                self.base_font = ("Consolas", 11)

                self._build_layout()
                self.html_text.tag_configure(
                    "error",
                    background="#5a1a1a",
                    foreground="#ffffff"
                )

            def _highlight_error(self, line):
                self.html_text.tag_remove("error", "1.0", tk.END)

                start = f"{line}.0"
                end = f"{line}.end"

                self.html_text.tag_add("error", start, end)
                self.html_text.see(start)


            def _build_layout(self):
                # Toolbar
                toolbar = ttk.Frame(self.root, padding=10)
                toolbar.pack(fill=tk.X)

                ttk.Button(toolbar, text="Run Parser", command=self.run_parser).pack(side=tk.LEFT)
                ttk.Button(toolbar, text="Clear Outputs", command=self.clear_outputs).pack(side=tk.LEFT, padx=(10, 0))

                # MAIN vertical pane (editor/tokens on top, output at bottom)
                vertical_pane = ttk.Panedwindow(self.root, orient=tk.VERTICAL)
                vertical_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # ─────────────── TOP: editor + tokens ───────────────
                top_pane = ttk.Panedwindow(vertical_pane, orient=tk.HORIZONTAL)

                # HTML editor
                editor_frame = ttk.Frame(top_pane)
                self._add_label(editor_frame, "HTML Source")
                self.html_text = self._create_scrolled_text(editor_frame)
                self.html_text.pack(fill=tk.BOTH, expand=True)
                top_pane.add(editor_frame, weight=3)

                # Tokens panel
                tokens_frame = ttk.Frame(top_pane)
                self._add_label(tokens_frame, "Tokens")
                self.tokens_text = self._create_scrolled_text(tokens_frame)
                self.tokens_text.pack(fill=tk.BOTH, expand=True)
                self.tokens_text.config(state=tk.DISABLED)
                top_pane.add(tokens_frame, weight=2)

                vertical_pane.add(top_pane, weight=4)

                # ─────────────── BOTTOM: terminal output ───────────────
                output_frame = ttk.Frame(vertical_pane)
                self._add_label(output_frame, "Parser Output (Terminal)")
                self.error_text = self._create_scrolled_text(output_frame, height=8)
                self.error_text.pack(fill=tk.BOTH, expand=True)
                self.error_text.config(state=tk.DISABLED)

                vertical_pane.add(output_frame, weight=1)

                # Default HTML
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
                self.html_text.tag_remove("error", "1.0", tk.END)

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
                    root = parser.parse()
                    semantic_errors = validate_semantics(root)
                    if semantic_errors:
                        for err in semantic_errors:
                            self._append_error(f"[SEMANTIC ERROR] {err}")
                        # Highlight the FIRST semantic error location
                        if semantic_errors[0]["line"] is not None:
                            self._highlight_error(semantic_errors[0]["line"])
                    else:
                        self._append_error("[SUCCESS] Parse complete. HTML is valid.")

                except ParseError as err:
                    self._append_error(f"[ERROR] {err}")
                    if err.line is not None:
                        self._highlight_error(err.line)
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
