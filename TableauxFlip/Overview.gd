extends VBoxContainer

class HistoryFrame:
	var clause_idx = -1
	var literal_idx = -1
	
	var previous_substitution_idx = -1
	var substitution_idx = -1
	var visited = []

var stack = []
var history = []
var substitutions = {}
var substitution_index = 0

var max_substitution_index = 0

var head = null

class Clause:
	var literal_list = []
	
	func _to_string():
		var string_literal_list = []
		
		for lit in literal_list:
			string_literal_list.append(str(lit))
		
		return " | ".join(PackedStringArray(literal_list))

class Literal:
	var negative = false
	var term
	
	var visual
	
	var clause_idx
	var literal_idx
	
	func _to_string():
		return to_string_with_substitutions({}, 0)
	
	func to_string_with_substitutions(substitutions, substitution_index):
		if negative:
			return "-" + term.to_string_with_substitutions(substitutions, substitution_index)
		return term.to_string_with_substitutions(substitutions, substitution_index)
	
	func unify(lit, substitutions, idx_from, idx_to):
		if lit.negative == negative:
			return false
		return term.unify(lit.term, substitutions, idx_from, idx_to)

class Term:
	pass

class Variable:
	var name = "Nameless"
	func _init(_name):
		name = _name
	
	func _to_string():
		return name
	
	func to_string_with_substitutions(substitutions, substitution_index):
		if [name, substitution_index] in substitutions:
			var sub = substitutions[[name, substitution_index]]
			return sub[0].to_string_with_substitutions(substitutions, sub[1])
		return name
	
	func unify(term, substitutions, idx_from, idx_to):
		while term is Variable and [term.name, idx_to] in substitutions:
			var sub = substitutions[[term.name, idx_to]]
			term = sub[0]
			idx_to = sub[1]
		
		if term is Variable:
			if term.name != name or idx_to != idx_from:
				substitutions[[term.name, idx_to]] = [self, idx_from]
			return true
		
		var this = self
		while this is Variable and [this.name, idx_from] in substitutions:
			var sub = substitutions[[this.name, idx_from]]
			this = sub[0]
			idx_from = sub[1]
		
		if this is Variable:
			substitutions[[this.name, idx_from]] = [term, idx_to]
			return true
		
		return this.unify(term, substitutions, idx_from, idx_to)

class Constant:
	var name = "Nameless"
	func _init(_name):
		name = _name
	
	func _to_string():
		return name
	
	func to_string_with_substitutions(_substitutions, _substitution_index):
		return name
	
	func unify(term, substitutions, idx_from, idx_to):
		while term is Variable and [term.name, idx_to] in substitutions:
			var sub = substitutions[[term.name, idx_to]]
			term = sub[0]
			idx_to = sub[1]
		
		if term is Variable:
			substitutions[[term.name, idx_to]] = [self, idx_from]
			return true
		
		if term is Constant and term.name == name:
			return true
		
		return false

class Application:
	var a = null
	var b = null
	func _init(_a,_b):
		a = _a
		b = _b
	
	func _to_string():
		var a_str = str(a)
		if a is Application:
			a_str = "(" + a_str + ")"
		var b_str = str(b)
		if b is Application:
			b_str = "(" + b_str + ")"
		return a_str + " " + b_str
	
	func to_string_with_substitutions(substitutions, substitution_index):
		var a_str = a.to_string_with_substitutions(substitutions, substitution_index)
		
		#if a_str.find(' ') != -1:
		#	a_str = "(" + a_str + ")"
		
		var b_str = b.to_string_with_substitutions(substitutions, substitution_index)
		if b_str.find(' ') != -1:
			b_str = "(" + b_str + ")"
		
		return a_str + " " + b_str
	
	func unify(term, substitutions, idx_from, idx_to):
		while term is Variable and [term.name, idx_to] in substitutions:
			var sub = substitutions[[term.name, idx_to]]
			term = sub[0]
			idx_to = sub[1]
		
		if term is Constant:
			return false
		
		if term is Application:
			if not a.unify(term.a, substitutions, idx_from, idx_to):
				return false
			return b.unify(term.b, substitutions, idx_from, idx_to)
		
		if term is Variable:
			substitutions[[term.name, idx_to]] = [self, idx_from]
			return true

var clause_list = []

func strip_parentheses(s):
	if s[0] == '(' and s[-1] == ')':
		s = s.substr(1, len(s)-2)
	return s

func parse_term(chunk):
	chunk = chunk.lstrip(' ').rstrip(' ')
	
	if len(chunk) == 0:
		return null
	
	if chunk.find(' ') == -1:
		if chunk[0] == '_':
			return Variable.new(chunk)
		else:
			return Constant.new(chunk)
	
	var subterms = []
	
	var depth = 0
	var split = 0
	
	for i in range(len(chunk)):
		if chunk[i] == '(':
			depth += 1
		if chunk[i] == ')':
			depth -= 1
		if chunk[i] == ' ' and depth == 0:
			var subterm = parse_term(strip_parentheses(chunk.substr(split, i - split)))
			
			if subterm == null:
				return null
			
			subterms.append(subterm)
			split = i + 1
	
	if depth != 0:
		return null
	
	subterms.append(parse_term(strip_parentheses(chunk.substr(split))))
	
	var result = Application.new(subterms[0], subterms[1])
	
	for i in range(2, len(subterms)):
		result = Application.new(result, subterms[i])
	
	return result

func parse_literal(chunk):
	chunk = chunk.lstrip(' ').rstrip(' ')
	if len(chunk) == 0:
		return null
	if chunk == '-':
		return null
	
	var result = Literal.new()
	
	if chunk[0] == '-': 
		result.negative = true
		chunk = chunk.substr(1)
	
	result.term = parse_term(chunk)
	
	if result.term == null:
		return null
	
	return result

func parse_clause(line, idx):
	var chunks = line.split('|')
	var clause = Clause.new()
	
	var lit_idx = 0
	for chunk in chunks:
		var literal = parse_literal(chunk)
		if literal == null:
			return null
		
		literal.clause_idx = idx
		literal.literal_idx = lit_idx
		
		lit_idx += 1
		
		clause.literal_list.append(literal)
	
	return clause

func parse():
	var new_clause_list = []
	
	var text = $Theorem/Input/Code.text
	var lines = text.split('\n')
	
	for line in lines:
		new_clause_list.append(parse_clause(line, len(new_clause_list)))
	
	for new_clause in new_clause_list:
		if new_clause == null:
			return
	
	for child in $Theorem/Commands/ClauseList.get_children():
		$Theorem/Commands/ClauseList.remove_child(child)
	
	var clause_index = 0
	for clause in new_clause_list:
		var c = preload("res://Clause.tscn").instantiate()
		c.clause = clause
		c.idx = clause_index
		
		clause_index += 1
		
		$Theorem/Commands/ClauseList.add_child(c)
		
		c.connect("literal_selected",Callable(self,"select_literal"))
	
	history = []
	clause_list = new_clause_list

func update_substitutions():
	for child in $Theorem/Commands/Stack.get_children():
		child.substitutions = substitutions
		
		var child_idx = child.get_index()
		
		if child_idx < len(stack):
			child.update_sub(try_reduce(child_idx))

func set_head(h):
	head = h
	
	var children = $Theorem/Commands/Stack.get_children()
	if len(children) < len(stack) or children[len(stack) - 1].substitution_idx != substitution_index:
		#print("Add new stack frame. Children is ", len(children), " and stack has ", len(stack), " elements.")
		if not len(children) < len(stack):
			print("Add new stack frame. Child sub idx is ", children[len(stack) - 1].substitution_idx, " and self sub idx is ", substitution_index, ".")
		
		var head_display = preload("res://StackFrame.tscn").instantiate()
		head_display.clause = clause_list[head.clause_idx]
		head_display.literal_idx = head.literal_idx
		
		head_display.substitutions = substitutions
		head_display.substitution_idx = substitution_index
		
		$Theorem/Commands/Stack.add_child(head_display)
		head_display.connect("gui_input",Callable(self,"on_reduce_gui_input").bind($Theorem/Commands/Stack.get_child_count() - 1))
	else:
		var child = children[len(stack)-1]
		var stack_frame = stack[-1]
		
		for i in range(len(stack_frame.visited)):
			if stack_frame.visited[i] == stack_frame.visited.max():
				child.literal_idx = i
	
	update_substitutions()
	
	# TODO: Set colors according to unifiability
	for clause_idx in len(clause_list):
		for literal_idx in len(clause_list[clause_idx].literal_list):
			var succes = try_extend(clause_idx, literal_idx)
			var visual = clause_list[clause_idx].literal_list[literal_idx].visual
			if succes:
				visual.modulate = Color.GREEN
			else:
				visual.modulate = Color.WHITE
	
	head.visual.modulate = Color.AQUA

func on_reduce_gui_input(event, index):
	if event.is_action_pressed("ui_select"):
		reduce(index)

func pop_stack():
	substitution_index = stack[-1].previous_substitution_idx
	stack.pop_back()
	
	if len(stack) != 0:
		if $Theorem/Commands/Stack.get_child_count() > len(stack):
			var child = $Theorem/Commands/Stack.get_child(len(stack))
			child.queue_free()
		
		var head = stack[-1]
		head.literal_idx += 1

func update_unifiability():
	for clause in clause_list:
		for literal in clause.literal_list:
			literal.visual.modulate = Color.WHITE

func jump():
	var unsolved_literal = -1
	while unsolved_literal == -1:
		var frame = stack[-1]
		for i in range(len(frame.visited)):
			if frame.visited[i] == -1:
				unsolved_literal = i
				
				var max_visit = stack[-1].visited.max() + 1
				stack[-1].visited[i] = max_visit
				
				set_head(clause_list[frame.clause_idx].literal_list[i])
				break
		
		if unsolved_literal == -1:
			pop_stack()
			
			if len(stack) == 0:
				print("SOLVED!")
				update_substitutions()
				update_unifiability()
				break

func try_extend(clause_idx, literal_idx):
	var substitutions_copy = substitutions.duplicate()
	return head.unify(clause_list[clause_idx].literal_list[literal_idx], substitutions_copy, substitution_index, max_substitution_index + 1)

func extend(clause_idx, literal_idx):
	var target = clause_list[clause_idx].literal_list[literal_idx]
	
	var frame = HistoryFrame.new()
	frame.clause_idx = clause_idx
	frame.literal_idx = literal_idx
	
	frame.previous_substitution_idx = substitution_index
	
	max_substitution_index += 1
	head.unify(target, substitutions, substitution_index, max_substitution_index)
	substitution_index = max_substitution_index
	
	frame.substitution_idx = substitution_index
	
	for i in range(len(clause_list[clause_idx].literal_list)):
		if i == literal_idx:
			frame.visited.append(0)
		else:
			frame.visited.append(-1)
	
	history.append(frame)
	stack.append(frame)
	
	jump()

func try_reduce(step_idx):
	var frame = stack[step_idx]
	var literal = clause_list[frame.clause_idx].literal_list[frame.literal_idx]
	
	return head.unify(literal, substitutions.duplicate(), substitution_index, frame.substitution_idx)

func reduce(step_idx):
	if try_reduce(step_idx):
		var frame = stack[step_idx]
		var literal = clause_list[frame.clause_idx].literal_list[frame.literal_idx]
		
		head.unify(literal, substitutions.duplicate(), substitution_index, frame.substitution_idx)
		jump()

func select_literal(clause_idx, literal_idx):
	if head == null:
		var frame = HistoryFrame.new()
		frame.clause_idx = clause_idx
		frame.literal_idx = literal_idx
		frame.substitution_idx = substitution_index
		
		for i in range(len(clause_list[clause_idx].literal_list)):
			if i == literal_idx:
				frame.visited.append(0)
			else:
				frame.visited.append(-1)
		
		history.append(frame)
		stack.append(frame)
		
		set_head(clause_list[clause_idx].literal_list[literal_idx])
	else:
		if clause_list[clause_idx].literal_list[literal_idx].visual.modulate == Color.GREEN:
			extend(clause_idx, literal_idx)


func restart():
	substitutions = {}
	substitution_index = 0
	head = null
	
	stack = []
	history = []
	
	for child in $Theorem/Commands/Stack.get_children():
		child.free()
	
	for clause in clause_list:
		for literal in clause.literal_list:
			literal.visual.modulate = Color.WHITE
