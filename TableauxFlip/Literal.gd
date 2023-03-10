extends HBoxContainer

signal literal_selected(clause_idx, literal_idx)

var clause
var idx = 0

var substitutions = {}

func _ready():
	$Counter.text = str(idx) + ": "
	
	for i in len(clause.literal_list):
		var lit = clause.literal_list[i]
		
		if i != 0:
			var label = Label.new()
			label.text = " | "
			add_child(label)
		
		var literal_scene = preload("res://Literal.tscn")
		var literal_instance = literal_scene.instantiate()
		
		literal_instance.text = str(lit)
		add_child(literal_instance)
		
		literal_instance.set_mouse_filter(Control.MOUSE_FILTER_STOP)
		literal_instance.connect("gui_input",Callable(self,"on_literal_gui_input").bind(i))
		
		lit.visual = literal_instance

func on_literal_gui_input(event, index):
	if event.is_action_pressed("ui_select"):
		on_literal_click(index)

func on_literal_click(lit_idx):
	emit_signal("literal_selected", idx, lit_idx)
