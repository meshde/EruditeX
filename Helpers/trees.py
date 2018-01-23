import numpy as np
from . import utils

class dt_node(object):
	def __init__(self, node, glove, children=[], dim=50):
		self.text = node.text
		self.pos_tag = node.pos_
		self.dep_tag = node.dep_
		self.head = node.head.text
		self.word_vector = utils.get_vector(node.text, glove, dim)
		self.hid_state = None
		self.children = children
		self.word_vector_size = dim
		self.count = None
		self.po_list = None

	def get_text(self):
		return self.text

	def get_children(self):
		return self.children

	def has_children(self):
		return not (len(self.children) == 0)

	def count_nodes(self):
		if self.count != None:
			return self.count
		
		count = 0
		if self.has_children():
			for cnode in self.children:
				count += cnode.count_nodes()
		
		self.count = 1 + count
		return self.count

	def postorder(self):
		if self.po_list:
			return self.po_list

		po_list = []

		if self.has_children():
			for cnode in self.children:
				for c in cnode.postorder():
					po_list.append(c)

		po_list.append(self)

		self.po_list = po_list
		return self.po_list

	def get_rnn_input(self):
		word_vector_list = self.get_tree_traversal('word_vector')
		parent_index_list = self.get_tree_traversal('parent_index')
		is_leaf_list = self.get_tree_traversal('is_leaf')
		dep_tag_list = self.get_tree_traversal('dep_tag')

		word_vector_array = np.array(word_vector_list).reshape((-1,self.word_vector_size))

		return word_vector_array,parent_index_list,is_leaf_list,dep_tag_list

	def get_tree_traversal(self,mode):
		postorder = self.postorder()
		node_list = []
		if mode == 'parent_index':
			node_list = []
			for node in postorder:
				count = 0
				for n in postorder:
					if n.text == node.head:
						node_list.append(count)
						break
					else:
						count += 1

		elif mode == 'text':
			node_list = [node.text for node in postorder]

		elif mode == 'word_vector':
			node_list = [node.word_vector for node in postorder]

		elif mode == 'is_leaf':
			node_list = [0 if node.has_children() else 1 for node in postorder]

		elif mode == 'dep_tag':
			dep_tags_dict = utils.load_dep_tags()
			node_list = [dep_tags_dict[node.dep_tag.upper()] for node in postorder]
		return node_list


class dtne_node(dt_node):
	def __init__(self, node, glove, children=[], dim=50):
		super().__init__(node, glove, children, dim)
		self.ent_type = utils.get_ne_index(node.ent_type)

	def get_tree_traversal(self, mode):
		postorder = self.postorder()
		if mode == 'ent_type':
			return [node.ent_type for node in postorder]
		return super().get_tree_traversal(mode)

	def get_rnn_input(self):
		inputs = super().get_rnn_input()
		ent_type = self.get_tree_traversal('ent_type')

		return inputs, ent_type