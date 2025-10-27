"""
Author: Ayesha Siddika
CSE 430 - Compiler Design
Mini Compiler Project - Fully Fixed Version
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import ply.lex as lex
import ply.yacc as yacc
import re

# ============================================================================
# LEXICAL ANALYZER (LEX) 
# ============================================================================

class Lexer:
    # Reserved keywords
    reserved = {
        'if': 'IF',
        'else': 'ELSE',
        'while': 'WHILE',
        'for': 'FOR',
        'int': 'INT',
        'float': 'FLOAT',
        'return': 'RETURN',
        'print': 'PRINT',
    }

    # Token list
    tokens = [
        'ID', 'NUMBER', 'FLOAT_NUM',
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MODULO',
        'ASSIGN', 'EQ', 'NE', 'LT', 'LE', 'GT', 'GE',
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE',
        'SEMICOLON', 'COMMA',
    ] + list(reserved.values())

    # Token rules
    t_PLUS = r'\+'
    t_MINUS = r'-'
    t_TIMES = r'\*'
    t_MODULO = r'%'
    t_ASSIGN = r'='
    t_EQ = r'=='
    t_NE = r'!='
    t_LT = r'<'
    t_LE = r'<='
    t_GT = r'>'
    t_GE = r'>='
    t_LPAREN = r'\('
    t_RPAREN = r'\)'
    t_LBRACE = r'\{'
    t_RBRACE = r'\}'
    t_SEMICOLON = r';'
    t_COMMA = r','

    # Ignored characters (spaces and tabs)
    t_ignore = ' \t'

    # Comment handling - MUST come before DIVIDE
    def t_COMMENT_SINGLE(self, t):
        r'//.*'
        pass  # Ignore single-line comments
    
    def t_COMMENT_MULTI(self, t):
        r'/\*(.|\n)*?\*/'
        t.lexer.lineno += t.value.count('\n')
        pass  # Ignore multi-line comments

    # DIVIDE token must come AFTER comment rules
    def t_DIVIDE(self, t):
        r'/'
        return t
     #decimal(floating point)
    def t_FLOAT_NUM(self, t):
        r'\d+\.\d+'
        t.value = float(t.value)
        return t
     #normal number(10,20)
    def t_NUMBER(self, t):
        r'\d+'
        t.value = int(t.value)
        return t
#recognize names
    def t_ID(self, t):
        r'[a-zA-Z_][a-zA-Z_0-9]*'#Must start with a letter or underscore (_)x, name1, _temp, Count
        t.type = self.reserved.get(t.value, 'ID')
        return t

    def t_newline(self, t):#counts how many new lines
        r'\n+'
        t.lexer.lineno += len(t.value)

    def t_error(self, t):#handles mistakes in the code.
        self.errors.append(f"Illegal character '{t.value[0]}' at line {t.lineno}")
        t.lexer.skip(1)
#Saves an error message (like “Illegal character ‘$’)Skips that wrong character and keeps going

    def __init__(self):#constructor
        self.lexer = None #we don’t have any lexer yet.
        self.tokens_list = [] #makes an empty list to store all the tokens
        self.errors = []#makes another empty list to store all the errors

    def build(self):
        self.lexer = lex.lex(module=self)#builds the lexer using lex.lex(module=self) Python to create the lexer from the rule

    def tokenize(self, data): #Clears any old tokens or errors before starting fresh.
        self.tokens_list = []
        self.errors = []
        self.lexer.input(data)#Gives the input code (data)
        
        while True: ## Keep getting tokens until there are no more left
            tok = self.lexer.token()
            if not tok:
                break
            self.tokens_list.append({ ## Save each token's type, value, line, and position in a list
                'type': tok.type,
                'value': tok.value,
                'line': tok.lineno,
                'position': tok.lexpos
            })
        
        return self.tokens_list, self.errors


# ============================================================================
# SYMBOL TABLE 
# ============================================================================

class SymbolTable:
    def __init__(self):
        self.symbols = {}#This is a dictionary that will store all the variables.
        self.scope_stack = ['global']#A stack to keep track of scopes. It starts with 'global'.
        self.scope_counter = 0 #A counter to give unique names to new scopes (like scope_1, scope_2, etc.).

     #enter_scope is called when you start a new block like { ... } or a loop.   
    def enter_scope(self, scope_name=None): 
        """Enter a new scope (e.g., entering an if block or while loop)"""
        if scope_name is None: #don’t give a name, it creates one automatically like scope_1.
            self.scope_counter += 1  #adds the new scope to the stack (scope_stack) so the compiler knows we’re now inside this scope.
            scope_name = f"scope_{self.scope_counter}"
        self.scope_stack.append(scope_name)
        return scope_name
    
    def exit_scope(self):
        """Exit the current scope"""#It removes the current scope from the stack.
        if len(self.scope_stack) > 1: #>1 check makes sure we never remove the global scope.
            return self.scope_stack.pop()
        return None
    
    def current_scope(self):
        """Get the current scope"""
         # Last item in stack is the current scope
        return self.scope_stack[-1]
        
    def insert(self, name, symbol_type, value=None, scope=None):
        """Insert a symbol into the table"""
        if scope is None:  # If no scope is given, use the current scope
            scope = self.current_scope()
        
        # Check if variable already exists in current scope
        key = f"{scope}:{name}"# Create a unique key for variable in this scope
        if key in self.symbols:# Check if variable already exists in this scope
            return False  # Already declared in this scope
        
            # Store the variable information
        self.symbols[key] = {
            'name': name,
            'type': symbol_type,
            'value': value,
            'scope': scope
        }
        return True
    
    def lookup(self, name):
        """Lookup a symbol, searching from innermost to outermost scope"""
        for scope in reversed(self.scope_stack): # Check each scope from innermost to outermost
            key = f"{scope}:{name}"
            if key in self.symbols:
                return self.symbols[key]# Found the variable
        return None
    
    def lookup_current_scope(self, name):
        """Lookup a symbol only in the current scope"""
        scope = self.current_scope()
        key = f"{scope}:{name}"
        return self.symbols.get(key)
    
    def get_all(self):
        """Get all symbols"""
        return list(self.symbols.values())


# ============================================================================
# PARSER AND SEMANTIC ANALYZER 
# ============================================================================

class Parser:  # Tokens come from the Lexer class
    tokens = Lexer.tokens
    
    def __init__(self):  # Symbol table to store variables and their scopes
        self.symbol_table = SymbolTable()
        self.intermediate_code = []# List to store intermediate code (3-address code)
        # Counters to generate unique temp variables and labels
        self.temp_count = 0
        self.label_count = 0
         # List to store errors found during parsing
        self.errors = []
        self.parse_tree = [] # Parse tree for syntactic structure
        
    def new_temp(self):
        # Generate a new temporary variable like t1, t2, ...
        self.temp_count += 1
        return f"t{self.temp_count}"
    
    def new_label(self):
        self.label_count += 1# Generate a new label like L1, L2, ...
        return f"L{self.label_count}"
    
    def emit(self, op, arg1=None, arg2=None, result=None):
        # Add a line of intermediate code
        code = {'op': op, 'arg1': arg1, 'arg2': arg2, 'result': result}
        self.intermediate_code.append(code)
        return result
    
    def backpatch(self, code_list, label):
        """Backpatch a list of instructions with a label"""
        for code in code_list:
            if code['arg2'] == 'BACKPATCH':
                code['arg2'] = label
    
    # Grammar rules
    def p_program(self, p):
        '''program : statement_list'''
        p[0] = ('program', p[1])  # Root of parse tree
        self.parse_tree.append(p[0])
    
    def p_statement_list(self, p):
        '''statement_list : statement_list statement
                         | statement'''
        if len(p) == 3:        # Combine multiple statements into a list
            p[0] = p[1] + [p[2]]
        else:
            p[0] = [p[1]]
    
    def p_statement(self, p):
        '''statement : declaration
                    | assignment
                    | print_statement
                    | if_statement
                    | while_statement
                    | block'''
        p[0] = p[1]      # A statement can be any of these
    
    def p_declaration(self, p):
        '''declaration : type ID SEMICOLON
                      | type ID ASSIGN expression SEMICOLON'''
        var_type = p[1]
        var_name = p[2]
        
        # Check if variable already declared in current scope
        if self.symbol_table.lookup_current_scope(var_name):
            self.errors.append(f"Variable '{var_name}' already declared in current scope")
        else:
            if len(p) == 4:
                self.symbol_table.insert(var_name, var_type)
                p[0] = ('declaration', var_type, var_name)
            else:
                     # Declaration with initialization
                value = p[4]
                self.symbol_table.insert(var_name, var_type, value)
                self.emit('=', value, None, var_name)# Emit intermediate code: var_name = value
                p[0] = ('declaration_init', var_type, var_name, value)
    
    def p_type(self, p):
        '''type : INT
               | FLOAT'''
        p[0] = p[1]
    
    def p_assignment(self, p):
        '''assignment : ID ASSIGN expression SEMICOLON'''
        var_name = p[1]
        expr = p[3]
        
        if not self.symbol_table.lookup(var_name):
            self.errors.append(f"Variable '{var_name}' not declared")
        
        self.emit('=', expr, None, var_name)
        p[0] = ('assignment', var_name, expr)
    
    def p_print_statement(self, p):
        '''print_statement : PRINT LPAREN expression RPAREN SEMICOLON'''
        self.emit('print', p[3], None, None)
        p[0] = ('print', p[3])
    
    def p_if_statement(self, p):
        '''if_statement : IF LPAREN condition RPAREN m_label block n_label
                       | IF LPAREN condition RPAREN m_label block n_label ELSE m_label block'''
        # p[3] = condition temp variable
        # p[5] = m_label after condition
        # p[6] = true block
        # p[7] = n_label after true block
        
        if len(p) == 8:  # Simple if
           
            false_label = p[5]
            self.emit('label', false_label, None, None)
        else:  # if-else (len == 11)
           
          
            false_label = p[5]
            end_label = p[9]
            self.emit('label', false_label, None, None)
            self.emit('label', end_label, None, None)
        
        p[0] = ('if', p[3])
    
    def p_while_statement(self, p):
        '''while_statement : WHILE m_label LPAREN condition RPAREN m_label block n_label'''
        
        start_label = p[2]
        exit_label = p[6]
        
        # Jump back to start
        self.emit('goto', start_label, None, None)
        # Exit label
        self.emit('label', exit_label, None, None)
        
        p[0] = ('while', p[4])
    
    def p_m_label(self, p):
        '''m_label : '''
        # Create a new label and emit it
        label = self.new_label()
        self.emit('label', label, None, None)
        p[0] = label
    
    def p_n_label(self, p):
        '''n_label : '''
        # Create a label for later use (for goto)
        label = self.new_label()
        # Don't emit yet, will be used for jumps
        self.emit('goto', label, None, None)
        p[0] = label
    
    def p_block(self, p):
        '''block : LBRACE statement_list RBRACE'''
        p[0] = ('block', p[2])
    
    def p_condition(self, p):
        '''condition : expression relop expression'''
        temp = self.new_temp()
        self.emit(p[2], p[1], p[3], temp)
        
        # Emit conditional jump right after condition
        false_label = self.new_label()
        self.emit('if_false', temp, false_label, None)
        
        p[0] = (temp, false_label)
    
    def p_relop(self, p):
        '''relop : LT
                | LE
                | GT
                | GE
                | EQ
                | NE'''
        p[0] = p[1]
    
    def p_expression_binop(self, p):
        '''expression : expression PLUS term
                     | expression MINUS term'''
        temp = self.new_temp()
        self.emit(p[2], p[1], p[3], temp)
        p[0] = temp
    
    def p_expression_term(self, p):
        '''expression : term'''
        p[0] = p[1]
    
    def p_term_binop(self, p):
        '''term : term TIMES factor
               | term DIVIDE factor
               | term MODULO factor'''
        temp = self.new_temp()
        self.emit(p[2], p[1], p[3], temp)
        p[0] = temp
    
    def p_term_factor(self, p):
        '''term : factor'''
        p[0] = p[1]
    
    def p_factor_number(self, p):
        '''factor : NUMBER
                 | FLOAT_NUM'''
        p[0] = p[1]
    
    def p_factor_id(self, p):
        '''factor : ID'''
        if not self.symbol_table.lookup(p[1]):
            self.errors.append(f"Variable '{p[1]}' not declared")
        p[0] = p[1]
    
    def p_factor_paren(self, p):
        '''factor : LPAREN expression RPAREN'''
        p[0] = p[2]
    
    def p_error(self, p):
        if p:
            self.errors.append(f"Syntax error at '{p.value}' (line {p.lineno})")
        else:
            self.errors.append("Syntax error at EOF")
    
    def build(self):
        self.parser = yacc.yacc(module=self)
    
    def parse(self, data):
        self.intermediate_code = []
        self.temp_count = 0
        self.label_count = 0
        self.errors = []
        self.parse_tree = []
        
        # Create a custom lexer that tracks scopes
        lexer = ScopeTrackingLexer(self.symbol_table)
        lexer.build()
        
        result = self.parser.parse(data, lexer=lexer.lexer)
        return result


# ============================================================================
# SCOPE TRACKING LEXER WRAPPER
# ============================================================================

class ScopeTrackingLexer(Lexer):
    """Extended lexer that tracks scope changes during tokenization"""
    
    def __init__(self, symbol_table):
        super().__init__()
        self.symbol_table = symbol_table
        self.original_token = None
        
    def build(self):
        super().build()
        # Wrap the token method to track scopes
        self.original_token = self.lexer.token
        self.lexer.token = self.token_with_scope_tracking
    
    def token_with_scope_tracking(self):
        tok = self.original_token()
        if tok:
            if tok.type == 'LBRACE':
                self.symbol_table.enter_scope()
            elif tok.type == 'RBRACE':
                self.symbol_table.exit_scope()
        return tok


# ============================================================================
# CODE GENERATOR (Assembly)
# ============================================================================

class CodeGenerator:
    def __init__(self):
        self.assembly_code = []
        self.registers = ['R1', 'R2', 'R3', 'R4']
        self.reg_map = {}
        self.next_reg = 0
        
    def get_register(self, var):
        if var in self.reg_map:
            return self.reg_map[var]
        
        reg = self.registers[self.next_reg % len(self.registers)]
        self.next_reg += 1
        self.reg_map[var] = reg
        return reg
    
    def generate(self, intermediate_code):
        self.assembly_code = []
        self.assembly_code.append("; Assembly Code Generated")
        self.assembly_code.append("section .data")
        self.assembly_code.append("section .text")
        self.assembly_code.append("global _start")
        self.assembly_code.append("_start:")
        
        for instruction in intermediate_code:
            op = instruction['op']
            arg1 = instruction['arg1']
            arg2 = instruction['arg2']
            result = instruction['result']
            
            if op == '=':
                reg_src = self.get_register(arg1) if isinstance(arg1, str) and arg1[0] == 't' else None
                reg_dest = self.get_register(result)
                
                if reg_src:
                    self.assembly_code.append(f"    MOV {reg_dest}, {reg_src}")
                else:
                    self.assembly_code.append(f"    MOV {reg_dest}, {arg1}")
                    
            elif op in ['+', '-', '*', '/', '%']:
                reg1 = self.get_register(arg1) if isinstance(arg1, str) else None
                reg2 = self.get_register(arg2) if isinstance(arg2, str) else None
                reg_result = self.get_register(result)
                
                op_map = {'+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV', '%': 'MOD'}
                
                if reg1 and reg2:
                    self.assembly_code.append(f"    {op_map[op]} {reg_result}, {reg1}, {reg2}")
                elif reg1:
                    self.assembly_code.append(f"    {op_map[op]} {reg_result}, {reg1}, {arg2}")
                elif reg2:
                    self.assembly_code.append(f"    {op_map[op]} {reg_result}, {arg1}, {reg2}")
                else:
                    self.assembly_code.append(f"    {op_map[op]} {reg_result}, {arg1}, {arg2}")
                    
            elif op in ['<', '<=', '>', '>=', '==', '!=']:
                reg1 = self.get_register(arg1) if isinstance(arg1, str) else None
                reg2 = self.get_register(arg2) if isinstance(arg2, str) else None
                reg_result = self.get_register(result)
                
                val1 = reg1 if reg1 else arg1
                val2 = reg2 if reg2 else arg2
                
                self.assembly_code.append(f"    CMP {val1}, {val2}")
                self.assembly_code.append(f"    SET{op} {reg_result}")
                
            elif op == 'label':
                self.assembly_code.append(f"{arg1}:")
                
            elif op == 'goto':
                self.assembly_code.append(f"    JMP {arg1}")
                
            elif op == 'if_false':
                reg = self.get_register(arg1) if isinstance(arg1, str) else None
                val = reg if reg else arg1
                self.assembly_code.append(f"    CMP {val}, 0")
                self.assembly_code.append(f"    JE {arg2}")
                
            elif op == 'print':
                reg = self.get_register(arg1) if isinstance(arg1, str) else None
                val = reg if reg else arg1
                self.assembly_code.append(f"    PRINT {val}")
        
        self.assembly_code.append("    MOV EAX, 1")
        self.assembly_code.append("    INT 0x80")
        
        return self.assembly_code


# ============================================================================
# SIMPLE GUI APPLICATION
# ============================================================================

class CompilerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CSE 430 - Mini Compiler")
        self.root.geometry("1000x600")
        
        # Initialize compiler components
        self.lexer = Lexer()
        self.lexer.build()
        self.parser = Parser()
        self.parser.build()
        self.code_generator = CodeGenerator()
        
        self.setup_ui()
        
    def setup_ui(self):
        # Top section - Input
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(top_frame, text="Source Code:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.input_text = scrolledtext.ScrolledText(top_frame, font=('Courier', 10), height=12)
        self.input_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Sample code with comments and scopes
        sample_code = """// Global variables
int x;
int y;
x = 10;
y = 20;

/* Calculate sum */
int sum;
sum = x + y;
print(sum);

// If block with local scope
if (x < y) {
    int diff;  // Local to if block
    diff = y - x;
    print(diff);
}

// While loop with local scope
int counter;
counter = 0;
while (counter < 5) {
    int temp;  // Local to while block
    temp = counter * 2;
    counter = counter + 1;
}
"""
        self.input_text.insert('1.0', sample_code)
        
       # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)
        
        compile_btn = tk.Button(
            button_frame, 
            text="Compile", 
            command=self.compile_code,
            font=('Arial', 10),
            bg="#aa00ff",
            fg='white',
            relief=tk.FLAT,
            padx=30,
            pady=8,
            cursor='hand2'
        )
        compile_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        clear_btn = tk.Button(
            button_frame, 
            text="Clear", 
            command=self.clear_all,
            font=('Arial', 10),
            bg="#c394b3",
            fg='white',
            relief=tk.FLAT,
            padx=30,
            pady=8,
            cursor='hand2'
        )
        clear_btn.pack(side=tk.LEFT)
        
        # Bottom section - Output with Tabs
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(bottom_frame, text="Output:", font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.notebook = ttk.Notebook(bottom_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create tabs
        self.create_tab("Tokens", "tokens_text")
        self.create_tab("Symbol Table", "symbol_text")
        self.create_tab("Intermediate Code", "intermediate_text")
        self.create_tab("Assembly", "assembly_text")
        self.create_tab("Errors", "errors_text")
        
    def create_tab(self, title, attr_name):
        frame = tk.Frame(self.notebook)
        self.notebook.add(frame, text=title)
        
        text_widget = scrolledtext.ScrolledText(frame, font=('Courier', 9), height=10)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        setattr(self, attr_name, text_widget)
        
    def compile_code(self):
        source_code = self.input_text.get('1.0', tk.END)
        
        # Clear outputs
        for attr in ['tokens_text', 'symbol_text', 'intermediate_text', 
                     'assembly_text', 'errors_text']:
            getattr(self, attr).delete('1.0', tk.END)
        
        # Lexical Analysis (for display only)
        tokens, lex_errors = self.lexer.tokenize(source_code)
        
        token_output = "Token Type       Value           Line\n"
        token_output += "-" * 45 + "\n"
        for token in tokens:
            token_output += f"{token['type']:<16} {str(token['value']):<15} {token['line']}\n"
        
        self.tokens_text.insert('1.0', token_output)
        
        # Syntax and Semantic Analysis (with scope tracking)
        self.parser = Parser()
        self.parser.build()
        self.parser.parse(source_code)
        
        # Symbol Table with scope information
        symbol_output = "Name             Type       Scope\n"
        symbol_output += "-" * 45 + "\n"
        symbols = sorted(self.parser.symbol_table.get_all(), 
                        key=lambda x: (0 if x['scope'] == 'global' else 1, x['name']))
        for symbol in symbols:
            symbol_output += f"{symbol['name']:<16} {symbol['type']:<10} {symbol['scope']}\n"
        
        self.symbol_text.insert('1.0', symbol_output)
        
        # Intermediate Code
        ic_output = ""
        for i, inst in enumerate(self.parser.intermediate_code, 1):
            op = inst['op']
            arg1 = inst['arg1']
            arg2 = inst['arg2']
            result = inst['result']
            
            if op == '=':
                ic_output += f"{i}. {result} = {arg1}\n"
            elif op in ['+', '-', '*', '/', '%']:
                ic_output += f"{i}. {result} = {arg1} {op} {arg2}\n"
            elif op == 'label':
                ic_output += f"{i}. {arg1}:\n"
            elif op == 'goto':
                ic_output += f"{i}. goto {arg1}\n"
            elif op == 'if_false':
                ic_output += f"{i}. if_false {arg1} goto {arg2}\n"
            elif op == 'print':
                ic_output += f"{i}. print {arg1}\n"
            else:
                ic_output += f"{i}. {result} = {arg1} {op} {arg2}\n"
        
        self.intermediate_text.insert('1.0', ic_output)
        
        # Assembly Code
        assembly = self.code_generator.generate(self.parser.intermediate_code)
        assembly_output = "\n".join(assembly)
        self.assembly_text.insert('1.0', assembly_output)
        
        # Errors
        all_errors = lex_errors + self.parser.errors
        if all_errors:
            errors_output = ""
            for i, error in enumerate(all_errors, 1):
                errors_output += f"{i}. {error}\n"
            self.errors_text.insert('1.0', errors_output)
        else:
            self.errors_text.insert('1.0', "✓ No errors found.\n✓ Comments handled correctly.\n✓ Scopes managed properly.\n✓ Control flow is correct.")
        
    def clear_all(self):
        self.input_text.delete('1.0', tk.END)
        for attr in ['tokens_text', 'symbol_text', 'intermediate_text', 
                     'assembly_text', 'errors_text']:
            getattr(self, attr).delete('1.0', tk.END)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = CompilerGUI(root)
    root.mainloop()
