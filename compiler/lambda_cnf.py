variables = {}
var_counter = 0
constants = {}
constant_counter = 0

merge_counter = 0

class LambdaTheorem:
    def __init__(self, code) -> None:
        self.clauses = []
        for line in code.split('\n'):
            self.clauses.append(LambdaClause(line))
    
    def flatten(self) -> None:
        new_clauses = []

        for i, clause in enumerate(self.clauses):
            clauses = clause.flatten()

            for new_clause in clauses:
                new_clauses.append(new_clause)

    def compile(self) -> str:
        return '\n'.join([clause.compile() for clause in self.clauses])

class LambdaClause:
    def __init__(self, code) -> None:
        if type(code) is list:
            self.literals = code
        else:
            self.literals = []
            for chunk in code.split('|'):
                chunk = chunk.strip()
                self.literals.append(LambdaLiteral(chunk))

    def flatten(self):
        if len(self.literals) <= 8:
            return [self]
        else:
            a = self.literals[0:7]
            b = self.literals[7:]

            var_intersect = list(a.variables.intersection(b.variables))

            term = LambdaConstant("_merge_" + str(merge_counter))
            merge_counter += 1

            for var in var_intersect:
                term = LambdaApplication(term, LambdaVariable(var))
            
            a.append

    def compile(self) -> str:
        compiled_literals = [literal.compile() for literal in self.literals]
        while len(compiled_literals) < 8:
            compiled_literals.append("")
        return ','.join(compiled_literals)

class LambdaLiteral:
    def __init__(self, code, negative=None) -> None:
        if negative == None:
            code = code.strip()
            self.negative = (code[0] != '-')
            code = code.strip('-')
            self.term = parse_term(code)
        else:
            self.negative = negative
            self.term = code
    
    def compile(self) -> str:
        return ('-' if self.negative else '') + self.term.compile()

    def unify(self, other, substitutions={}):
        pass

def truncate_parentheses(s) -> str:
    if s[0] == '(':
        return s[1:-1]
    return s

def parse_term(term_string):
    term_string = term_string.strip()

    if term_string.find(' ') == -1:
        if term_string[0] == '_':
            return LambdaVariable(term_string)
        else:
            return LambdaConstant(term_string)
    
    depth = 0
    separator = 0

    split = []

    for i in range(len(term_string)):
        if term_string[i] == '(':
            depth += 1
        if term_string[i] == ')':
            depth -= 1
        
        if depth == 0 and term_string[i] == ' ':
            split.append(term_string[separator:i])
            separator = i+1
    
    split.append(term_string[separator:])
        
    assert(depth == 0)

    parsed_split = [parse_term(truncate_parentheses(c)) for c in split]
    result = LambdaApplication(parsed_split[0], parsed_split[1])
    for j in range(2, len(parsed_split)):
        result = LambdaApplication(result, parsed_split[j])

    return result

class LambdaTerm:
    def __init__(self, code) -> None:
        assert(False)

class LambdaConstant(LambdaTerm):
    def __init__(self, code) -> None:
        self.name = code
        self.variables = set()
    
    def compile(self) -> str:
        global constants
        global constant_counter

        if not self.name in constants:
            constants[self.name] = "0xC{:07d}".format(constant_counter)
            constant_counter += 1
        
        return constants[self.name]

class LambdaVariable(LambdaTerm):
    def __init__(self, code) -> None:
        self.name = code
        self.variables = {self.name}

    def compile(self) -> str:
        global variables
        global var_counter

        if not self.name in variables:
            variables[self.name] = "0x8{:07d}".format(var_counter)
        
        var_counter += 1

        return variables[self.name]

class LambdaApplication(LambdaTerm):
    def __init__(self, a, b) -> None:
        #self.a = parse_term(a)
        #self.b = parse_term(b)
        self.a = a
        self.b = b

        self.variables = self.a.variables.union(self.b.variables)


    def compile(self) -> str:
        return "(%s, %s)" % (self.a.compile(), self.b.compile())

def compile_lambda(theorem):
    lambda_theorem = LambdaTheorem(theorem)

    return lambda_theorem