#!/usr/bin/env /usr/bin/python3

import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import (
							FigureCanvasQTAgg as FigureCanvas,
							NavigationToolbar2QT as NavigationToolbar
							)
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors, ticker, cm
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import Delaunay, Voronoi, ConvexHull
from trimesh import Trimesh
from trimesh.repair import fix_normals
from trimesh.graph import connected_components
from trimesh.smoothing import filter_humphrey
import mahotas as mh
from PyQt5.QtCore import Qt, QPoint, QRect, QSize
from PyQt5.QtGui import QIntValidator, QMouseEvent
from PyQt5.QtWidgets import (
							QApplication, QLabel, QWidget,
							QPushButton, QHBoxLayout, QVBoxLayout,
							QComboBox, QCheckBox, QSlider, QProgressBar,
							QFormLayout, QLineEdit, QTabWidget,
							QSizePolicy, QFileDialog, QMessageBox
							)
from pathlib import Path
from aicsimageio import AICSImage
#from aicspylibczi import CziFile
#from nd2reader import ND2Reader

################################################################################
# colormaps for matplotlib #
############################

red_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'alpha': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
			}
red_cmap = LinearSegmentedColormap('red_cmap', red_cdict)
cm.register_cmap(cmap=red_cmap)

green_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'alpha': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
			}
green_cmap = LinearSegmentedColormap('green_cmap', green_cdict)
cm.register_cmap(cmap=green_cmap)

blue_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 0.0, 0.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'alpha': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
			}
blue_cmap = LinearSegmentedColormap('blue_cmap', blue_cdict)
cm.register_cmap(cmap=blue_cmap)

transparent_cdict = {
		'red':   ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'green': ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'blue':  ((0, 0.0, 0.0),
				  (1, 1.0, 1.0)),
		'alpha': ((0, 1.0, 1.0),
				  (1, 0.0, 0.0)),
			}
transparent_cmap = LinearSegmentedColormap('transparent_cmap',
											transparent_cdict)
cm.register_cmap(cmap=transparent_cmap)

################################################################################
# class for triangulation #
###########################

class SimplicialComplex ():
	def __init__ (self, points = None,
						simplices = None,
						neighbours = None):
		self.points = points
		self.simplices = simplices
		self.neighbours = neighbours
		self.longest_edges = None
		if simplices is not None:
			self.calc_longest_edges()
	
	def calc_longest_edges (self):
	#	self.longest_edges = np.array(self.simplices.shape[0], dtype = float)
		point_array = self.points[self.simplices]
		self.longest_edges = np.amax(np.linalg.norm(
						point_array[:,np.newaxis,:,:] - \
						point_array[:,:,np.newaxis,:],
							axis=-1),axis=(1,2))
	
	def remove_simplex (self, index):
		self.neighbours[self.neighbours == index] = -1
		self.neighbours[self.neighbours > index] -= 1
		self.simplices = np.delete(self.simplices, index, axis=0)
		self.longest_edges = np.delete(self.longest_edges, index)
	
	def remove_long_simplices (self, length):
		for index in np.arange(self.simplices.shape[0]-1,-1,-1):
			if self.longest_edges[index] > length:
				self.remove_simplex(index)

################################################################################
# function to quickly calclate shortest distance to line segment #
##################################################################

def lineseg_dists (p, a, b):
	# Handle case where p is a single point, i.e. 1d array.
	p = np.atleast_2d(p)
	# possibly faster norms with numba
	if np.all(a == b):
		return np.linalg.norm(p - a, axis=1)
	# normalized tangent vector
	d = np.divide(b - a, np.linalg.norm(b - a))
	# signed parallel distance components
	s = np.dot(a - p, d)
	t = np.dot(p - b, d)
	# clamped parallel distance
	h = np.maximum.reduce([s, t, np.zeros(len(p))])
	# perpendicular distance component, as before
	# note that for the 3D case these will be vectors
	c = np.cross(p - a, d)
	# use hypot for Pythagoras to improve accuracy
	return np.hypot(h, c)

def display_error (error_text = 'Something went wrong!'):
	msg = QMessageBox()
	msg.setIcon(QMessageBox.Critical)
	msg.setText("Error")
	msg.setInformativeText(error_text)
	msg.setWindowTitle("Error")
	msg.exec_()

################################################################################
# canvas widget to put matplotlib plot #
########################################

class MPLCanvas (FigureCanvas):
	def __init__ (self, parent=None, width=8, height=8, dpi=100):
		self.fig = Figure(figsize=(width, height), dpi=dpi)
		self.ax = self.fig.add_subplot(111)
		FigureCanvas.__init__(self, self.fig)
		self.setParent(parent)
		FigureCanvas.setSizePolicy(self,
				QSizePolicy.Expanding,
				QSizePolicy.Expanding)
		FigureCanvas.updateGeometry(self)
		self.fig.tight_layout()
		self.dapi_image = np.ones((512,512),dtype=int)
		self.green_image = np.zeros((512,512),dtype=int)
		self.red_image = np.zeros((512,512),dtype=int)
		self.box = np.array([[0,512], [0,512]])
		self.dapi_plot = None
		self.green_plot = None
		self.red_plot = None
		self.box_plot = None
		self.show_green = True
		self.show_red = True
		self.show_box = False
		self.show_mesh = False
		self.select_box = None
		self.dapi_centres = np.zeros((0,2), dtype = float)
		self.green_cells = np.zeros((0,1), dtype = bool)
		self.red_cells = np.zeros((0,1), dtype = bool)
		self.epi_cells = np.zeros((0,1), dtype = bool)
		self.edges = np.zeros((0,2), dtype = int)
		self.edges_outer = np.zeros((0,1), dtype = bool)
		self.edges_outer_red = np.zeros((0,1), dtype = bool)
		self.edges_outer_green = np.zeros((0,1), dtype = bool)
		self.dapi_centres_plot = None
		self.green_centres_plot = None
		self.red_centres_plot = None
		self.epi_centres_plot = None
		self.edges_plot = None
		self.edges_outer_plot = None
		self.edges_outer_red_plot = None
		self.edges_outer_green_plot = None
		self.edges_outer_purple_plot = None
		self.points_scale = 1.0
		self.plot()
	
	def update_images (self, dapi_image, green_image, red_image,
						show_green = True, show_red = True,
						box = np.array([[0,512], [0,512]]),
						show_box = False, show_mesh = False):
		self.dapi_image = dapi_image
		self.green_image = green_image
		self.red_image = red_image
		self.show_green = show_green
		self.show_red = show_red
		self.box = box
		self.show_box = show_box
		self.show_mesh = show_mesh
		if dapi_image is not None:
			self.points_scale = np.array(dapi_image.shape).astype(float)[0] \
									/ 500.
		self.plot()
	
	def update_centres (self, dapi_centres,
							green_cells = None, red_cells = None,
							epi_cells = None, edges = None, edges_outer = None,
							edges_outer_red = None, edges_outer_green = None):
		self.dapi_centres = dapi_centres
		self.green_cells = green_cells
		self.red_cells = red_cells
		self.epi_cells = epi_cells
		self.edges = edges
		self.edges_outer = edges_outer
		self.edges_outer_red = edges_outer_red
		self.edges_outer_green = edges_outer_green
		self.plot()
	
	def plot (self):
		if self.dapi_plot is not None:
			self.dapi_plot.remove()
		if self.green_plot is not None:
			self.green_plot.remove()
		if self.red_plot is not None:
			self.red_plot.remove()
		# plots
		self.ax.set_xlim(left = 0, right = len(self.dapi_image[0,:]))
		self.ax.set_ylim(bottom = 0, top = len(self.dapi_image[:,0]))
		self.dapi_plot = self.ax.imshow(self.dapi_image ,cmap=transparent_cmap)
		if self.show_green:
			self.green_plot = self.ax.imshow(self.green_image ,cmap=green_cmap)
		else:
			self.green_plot = None
		if self.show_red:
			self.red_plot = self.ax.imshow(self.red_image ,cmap=red_cmap)
		else:
			self.red_plot = None
		self.plot_box()
		self.plot_centres()
	#	self.fig.canvas.draw_idle()
		self.draw()
	
	def plot_box (self):
		self.remove_box()
		if self.show_box:
			self.box_plot = self.ax.plot((self.box[0,0], self.box[0,1],
											self.box[0,1], self.box[0,0],
												self.box[0,0]),
										 (self.box[1,0], self.box[1,0],
											self.box[1,1], self.box[1,1],
												self.box[1,0]),
										 color='gold', linestyle='-')
		else:
			self.box_plot = None
	
	def remove_box (self):
		if self.box_plot is not None:
			if isinstance(self.box_plot,list):
				for line in self.box_plot:
					line.remove()
			else:
			#	self.box_plot.remove()
				self.box_plot = None
	
	def plot_centres (self):
		self.remove_centres()
		scale = (800./(self.dapi_image.shape[0]) + \
				 800./(self.dapi_image.shape[1])) * self.points_scale
		if self.dapi_centres.shape[0] > 0:
			if self.epi_cells is not None:
				if self.epi_cells.shape[0] > 0:
					self.epi_centres_plot = self.ax.plot(
									self.dapi_centres[self.epi_cells,0],
									self.dapi_centres[self.epi_cells,1],
									color = 'royalblue', linestyle = '', 
									marker = 'o', markersize = scale*1.7)
			self.dapi_centres_plot = self.ax.plot(
									self.dapi_centres[:,0],
									self.dapi_centres[:,1],
									color = 'white', linestyle = '', 
									marker = 'o', markersize = scale)
			if (self.edges is not None) and self.show_mesh:
				if self.edges.shape[0] > 0:
					line_collection_edges = LineCollection(
								self.dapi_centres[self.edges],
									colors = 'white')
					self.edges_plot = self.ax.add_collection(
														line_collection_edges)
			if self.edges_outer is not None:
				if self.edges_outer.shape[0] > 0:
					line_collection_outer = LineCollection(
								self.dapi_centres[self.edges[self.edges_outer]],
									colors = 'royalblue')
					self.edges_outer_plot = self.ax.add_collection(
														line_collection_outer)
			if self.edges_outer_red is not None:
				if self.edges_outer_red.shape[0] > 0:
					line_collection_outer_red = LineCollection(
						self.dapi_centres[self.edges[self.edges_outer_red]],
									colors = 'crimson')
					self.edges_outer_red_plot = self.ax.add_collection(
												line_collection_outer_red)
			if self.edges_outer_green is not None:
				if self.edges_outer_green.shape[0] > 0:
					line_collection_outer_green = LineCollection(
						self.dapi_centres[self.edges[self.edges_outer_green]],
									colors = 'seagreen')
					self.edges_outer_green_plot = self.ax.add_collection(
												line_collection_outer_green)
			if (self.edges_outer_red is not None) and \
			   (self.edges_outer_green is not None) :
				if (self.edges_outer_red.shape[0] > 0) and \
				   (self.edges_outer_green.shape[0] > 0):
					line_collection_outer_purple = LineCollection(
						self.dapi_centres[self.edges[self.edges_outer_red & \
													 self.edges_outer_green]],
									colors = 'seagreen')
					self.edges_outer_purple_plot = self.ax.add_collection(
												line_collection_outer_purple)
			if self.red_cells is not None:
				if self.red_cells.shape[0] > 0:
					self.red_centres_plot = self.ax.plot(
									self.dapi_centres[self.red_cells,0],
									self.dapi_centres[self.red_cells,1],
									color = 'crimson', linestyle = '', 
									marker = '+', markersize = scale*1.3)
			if self.green_cells is not None:
				if self.green_cells.shape[0] > 0:
					self.green_centres_plot = self.ax.plot(
									self.dapi_centres[self.green_cells,0],
									self.dapi_centres[self.green_cells,1],
									color = 'seagreen', linestyle = '',
									marker = 'x', markersize = scale)
	
	def remove_plot_element (self, plot_element):
		if isinstance(plot_element,list):
			for line in plot_element:
				line.remove()
		else:
			plot_element.remove()
	
	def remove_centres (self):
		if self.dapi_centres_plot is not None:
			self.remove_plot_element(self.dapi_centres_plot)
			self.dapi_centres_plot = None
		if self.edges_plot is not None:
			self.remove_plot_element(self.edges_plot)
			self.edges_plot = None
		if self.edges_outer_plot is not None:
			self.remove_plot_element(self.edges_outer_plot)
			self.edges_outer_plot = None
		if self.edges_outer_red_plot is not None:
			self.remove_plot_element(self.edges_outer_red_plot)
			self.edges_outer_red_plot = None
		if self.edges_outer_green_plot is not None:
			self.remove_plot_element(self.edges_outer_green_plot)
			self.edges_outer_green_plot = None
		if self.edges_outer_purple_plot is not None:
			self.remove_plot_element(self.edges_outer_purple_plot)
			self.edges_outer_purple_plot = None
		if self.green_centres_plot is not None:
			self.remove_plot_element(self.green_centres_plot)
			self.green_centres_plot = None
		if self.red_centres_plot is not None:
			self.remove_plot_element(self.red_centres_plot)
			self.red_centres_plot = None
		if self.epi_centres_plot is not None:
			self.remove_plot_element(self.epi_centres_plot)
			self.epi_centres_plot = None
	
	def plot_selector (self, p_1, p_2):
		self.remove_selector()
		self.select_box = self.ax.plot((p_1[0], p_2[0], p_2[0], p_1[0],
										p_1[0]),
									   (p_1[1], p_1[1], p_2[1], p_2[1],
										p_1[1]),
									  color = 'white',
									  linestyle = '-')
		self.draw()
	
	def remove_selector (self):
		if self.select_box:
			if isinstance(self.select_box,list):
				for line in self.select_box:
					line.remove()
			self.select_box = None

################################################################################
# main window widget #
######################

class Window (QWidget):
	def __init__ (self):
		super().__init__()
		self.green_active = True
		self.red_active = True
		self.geometry_active = True
		self.green_cutoff_active = False
		self.red_cutoff_active = False
		self.threshold_defaults = np.array([180,2000,4095,4095,
											320,2000,4095,4095,
											50,40,10])
		self.green_lower = self.threshold_defaults[0]
		self.green_upper = self.threshold_defaults[1]
		self.green_cutoff = self.threshold_defaults[2]
		self.green_max = self.threshold_defaults[3]
		self.red_lower = self.threshold_defaults[4]
		self.red_upper = self.threshold_defaults[5]
		self.red_cutoff = self.threshold_defaults[6]
		self.red_max = self.threshold_defaults[7]
		self.geo_edge_max = self.threshold_defaults[8]
		self.geo_distance = self.threshold_defaults[9]
		self.geo_dist_red = self.threshold_defaults[10]
		self.geo_size = int(512/8)
		self.x_lower = 0
		self.x_upper = 0
		self.x_size = 512
		self.y_lower = 0
		self.y_upper = 0
		self.y_size = 512
		self.z_level = 0
		self.z_size = 1
		self.z_lower = 0
		self.z_upper = 0
		self.zoomed = False
		self.dapi_image = np.ones((512,512),dtype=int)
		self.green_image = np.zeros((512,512),dtype=int)
		self.red_image = np.zeros((512,512),dtype=int)
		self.file_path = None
		self.image_stack = None
		self.channel_names = None
		self.title = "Nuclear Position Tool"
		self.canvas = MPLCanvas()
		self.toolbar = NavigationToolbar(self.canvas, self)
		self.selecting_area = False
		self.click_id = 0
		self.move_id = 0
		self.position = np.array([0,0])
		self.advanced_defaults = np.array([9,1,4,2,6,4])
		self.neighbourhood_size = self.advanced_defaults[0]
		self.threshold_difference = self.advanced_defaults[1]
		self.minimum_distance = self.advanced_defaults[2]
		self.gauss_deviation = self.advanced_defaults[3]
		self.max_layer_distance = self.advanced_defaults[4]
		self.number_layer_cell = self.advanced_defaults[5]
		self.scale = np.array([0.232, 0.232, 0.479])
		self.dapi_centres = np.zeros((0,2), dtype = float)
		self.green_cells = np.zeros((0,1), dtype = bool)
		self.red_cells = np.zeros((0,1), dtype = bool)
		self.epi_cells = np.zeros((0,1), dtype = bool)
		self.edges = np.zeros((0,2), dtype = int)
		self.edges_outer = np.zeros((0,1), dtype = bool)
		self.edges_outer_red = np.zeros((0,1), dtype = bool)
		self.edges_outer_green = np.zeros((0,1), dtype = bool)
		self.mesh = None
		self.plot_mesh = False
		self.plot_dapi = False
		#
		self.setupGUI()
	
	def setupGUI (self):
		self.setWindowTitle(self.title)
		# layout for full window
		outer_layout = QVBoxLayout()
		# top section for plot and sliders
		main_layout = QHBoxLayout()
		# main left for plot
		plot_layout = QVBoxLayout()
		plot_layout.addWidget(self.canvas)
		toolbar_layout = QHBoxLayout()
		toolbar_layout.addWidget(self.toolbar)
		toolbar_layout.addWidget(QLabel('Z:'))
		self.textbox_z = QLineEdit()
		self.textbox_z.setMaxLength(4)
		self.textbox_z.setFixedWidth(50)
		self.textbox_z.setText(str(self.z_level))
		self.textbox_z.setValidator(QIntValidator())
		self.textbox_z.editingFinished.connect(self.z_textbox_select)
		toolbar_layout.addWidget(self.textbox_z)
		self.button_z_min = QPushButton()
		self.button_z_min.setText('Set Z Min')
		self.button_z_min.clicked.connect(self.z_min_button)
		toolbar_layout.addWidget(self.button_z_min)
		self.button_z_max = QPushButton()
		self.button_z_max.setText('Set Z Max')
		self.button_z_max.clicked.connect(self.z_max_button)
		toolbar_layout.addWidget(self.button_z_max)
		plot_layout.addLayout(toolbar_layout)
		main_layout.addLayout(plot_layout)
		# main right for options
		options_layout = QHBoxLayout()
		z_select_layout = QVBoxLayout()
		self.slider_z = QSlider(Qt.Vertical)
		self.setup_z_slider()
		self.slider_z.valueChanged.connect(self.z_slider_select)
		z_select_layout.addWidget(self.slider_z)
		options_layout.addLayout(z_select_layout)
		tabs = QTabWidget()
		tabs.setMinimumWidth(220)
		tabs.setMaximumWidth(220)
		# green channel options tab
		tab_green = QWidget()
		tab_green.layout = QVBoxLayout()
		# checkbox to turn off green channel
		self.checkbox_green = QCheckBox("green channel active")
		self.checkbox_green.setChecked(self.green_active)
		self.checkbox_green.stateChanged.connect(self.green_checkbox)
		tab_green.layout.addWidget(self.checkbox_green)
		#checkbox to turn on green cutoff feature
		self.checkbox_green_cutoff = QCheckBox("green cutoff active")
		self.checkbox_green_cutoff.setChecked(self.green_cutoff_active)
		self.checkbox_green_cutoff.stateChanged.connect(
												self.green_cutoff_checkbox)
		tab_green.layout.addWidget(self.checkbox_green_cutoff)
		# sliders for green thresholds
		threshold_layout_green = QHBoxLayout()
		# green min
		threshold_layout_green_min = QVBoxLayout()
		slider_layout_green_min = QHBoxLayout()
		self.slider_green_min = QSlider(Qt.Vertical)
		self.slider_green_min.valueChanged.connect(self.threshold_green_lower)
		slider_layout_green_min.addWidget(self.slider_green_min)
		label_green_min = QLabel('lower')
		label_green_min.setAlignment(Qt.AlignCenter)
		self.textbox_green_min = QLineEdit()
		self.textbox_green_min.setMaxLength(4)
		self.textbox_green_min.setFixedWidth(50)
		self.textbox_green_min.setValidator(QIntValidator())
		self.textbox_green_min.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_green_min.addLayout(slider_layout_green_min)
		threshold_layout_green_min.addWidget(label_green_min)
		threshold_layout_green_min.addWidget(self.textbox_green_min)
		# green max
		threshold_layout_green_max = QVBoxLayout()
		slider_layout_green_max = QHBoxLayout()
		self.slider_green_max = QSlider(Qt.Vertical)
		self.slider_green_max.valueChanged.connect(self.threshold_green_upper)
		slider_layout_green_max.addWidget(self.slider_green_max)
		label_green_max = QLabel('upper')
		label_green_max.setAlignment(Qt.AlignCenter)
		self.textbox_green_max = QLineEdit()
		self.textbox_green_max.setMaxLength(4)
		self.textbox_green_max.setFixedWidth(50)
		self.textbox_green_max.setValidator(QIntValidator())
		self.textbox_green_max.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_green_max.addLayout(slider_layout_green_max)
		threshold_layout_green_max.addWidget(label_green_max)
		threshold_layout_green_max.addWidget(self.textbox_green_max)
		# green cutoff
		threshold_layout_green_cut = QVBoxLayout()
		slider_layout_green_cut = QHBoxLayout()
		self.slider_green_cut = QSlider(Qt.Vertical)
		self.slider_green_cut.valueChanged.connect(self.threshold_green_cutoff)
		slider_layout_green_cut.addWidget(self.slider_green_cut)
		label_green_cut = QLabel('cutoff')
		label_green_cut.setAlignment(Qt.AlignCenter)
		self.textbox_green_cut = QLineEdit()
		self.textbox_green_cut.setMaxLength(4)
		self.textbox_green_cut.setFixedWidth(50)
		self.textbox_green_cut.setValidator(QIntValidator())
		self.textbox_green_cut.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_green_cut.addLayout(slider_layout_green_cut)
		threshold_layout_green_cut.addWidget(label_green_cut)
		threshold_layout_green_cut.addWidget(self.textbox_green_cut)
		#
		threshold_layout_green.addLayout(threshold_layout_green_min)
		threshold_layout_green.addLayout(threshold_layout_green_max)
		threshold_layout_green.addLayout(threshold_layout_green_cut)
		tab_green.layout.addLayout(threshold_layout_green)
		tab_green.setLayout(tab_green.layout)
		tabs.addTab(tab_green, 'green')
		# red channel options tab
		tab_red = QWidget()
		tab_red.layout = QVBoxLayout()
		# checkbox to turn off red channel
		self.checkbox_red = QCheckBox("red channel active")
		self.checkbox_red.setChecked(self.red_active)
		self.checkbox_red.stateChanged.connect(self.red_checkbox)
		tab_red.layout.addWidget(self.checkbox_red)
		#checkbox to turn on red cutoff feature
		self.checkbox_red_cutoff = QCheckBox("red cutoff active")
		self.checkbox_red_cutoff.setChecked(self.red_cutoff_active)
		self.checkbox_red_cutoff.stateChanged.connect(
												self.red_cutoff_checkbox)
		tab_red.layout.addWidget(self.checkbox_red_cutoff)
		# sliders for red thresholds
		threshold_layout_red = QHBoxLayout()
		# red min
		threshold_layout_red_min = QVBoxLayout()
		slider_layout_red_min = QHBoxLayout()
		self.slider_red_min = QSlider(Qt.Vertical)
		self.slider_red_min.valueChanged.connect(self.threshold_red_lower)
		slider_layout_red_min.addWidget(self.slider_red_min)
		label_red_min = QLabel('lower')
		label_red_min.setAlignment(Qt.AlignCenter)
		self.textbox_red_min = QLineEdit()
		self.textbox_red_min.setMaxLength(4)
		self.textbox_red_min.setFixedWidth(50)
		self.textbox_red_min.setValidator(QIntValidator())
		self.textbox_red_min.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_red_min.addLayout(slider_layout_red_min)
		threshold_layout_red_min.addWidget(label_red_min)
		threshold_layout_red_min.addWidget(self.textbox_red_min)
		# red max
		threshold_layout_red_max = QVBoxLayout()
		slider_layout_red_max = QHBoxLayout()
		self.slider_red_max = QSlider(Qt.Vertical)
		self.slider_red_max.valueChanged.connect(self.threshold_red_upper)
		slider_layout_red_max.addWidget(self.slider_red_max)
		label_red_max = QLabel('upper')
		label_red_max.setAlignment(Qt.AlignCenter)
		self.textbox_red_max = QLineEdit()
		self.textbox_red_max.setMaxLength(4)
		self.textbox_red_max.setFixedWidth(50)
		self.textbox_red_max.setValidator(QIntValidator())
		self.textbox_red_max.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_red_max.addLayout(slider_layout_red_max)
		threshold_layout_red_max.addWidget(label_red_max)
		threshold_layout_red_max.addWidget(self.textbox_red_max)
		# red cutoff
		threshold_layout_red_cut = QVBoxLayout()
		slider_layout_red_cut = QHBoxLayout()
		self.slider_red_cut = QSlider(Qt.Vertical)
		self.slider_red_cut.valueChanged.connect(self.threshold_red_cutoff)
		slider_layout_red_cut.addWidget(self.slider_red_cut)
		label_red_cut = QLabel('cutoff')
		label_red_cut.setAlignment(Qt.AlignCenter)
		self.textbox_red_cut = QLineEdit()
		self.textbox_red_cut.setMaxLength(4)
		self.textbox_red_cut.setFixedWidth(50)
		self.textbox_red_cut.setValidator(QIntValidator())
		self.textbox_red_cut.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_red_cut.addLayout(slider_layout_red_cut)
		threshold_layout_red_cut.addWidget(label_red_cut)
		threshold_layout_red_cut.addWidget(self.textbox_red_cut)
		#
		threshold_layout_red.addLayout(threshold_layout_red_min)
		threshold_layout_red.addLayout(threshold_layout_red_max)
		threshold_layout_red.addLayout(threshold_layout_red_cut)
		tab_red.layout.addLayout(threshold_layout_red)
		tab_red.setLayout(tab_red.layout)
		tabs.addTab(tab_red, 'red')
		# geometry analysis options tab
		tab_geo = QWidget()
		tab_geo.layout = QVBoxLayout()
		# checkbox to turn off geometry analysis
		self.checkbox_geo = QCheckBox("geometry analysis")
		self.checkbox_geo.setChecked(self.geometry_active)
		self.checkbox_geo.stateChanged.connect(self.geo_checkbox)
		tab_geo.layout.addWidget(self.checkbox_geo)
		# sliders for geometry thresholds
		threshold_layout_geo = QHBoxLayout()
		# geometry max edge length
		threshold_layout_geo_max = QVBoxLayout()
		slider_layout_geo_max = QHBoxLayout()
		self.slider_geo_max = QSlider(Qt.Vertical)
		self.slider_geo_max.valueChanged.connect(self.threshold_geo_max)
		slider_layout_geo_max.addWidget(self.slider_geo_max)
		label_geo_max = QLabel('len_edge')
		label_geo_max.setAlignment(Qt.AlignCenter)
		self.textbox_geo_max = QLineEdit()
		self.textbox_geo_max.setMaxLength(4)
		self.textbox_geo_max.setFixedWidth(50)
		self.textbox_geo_max.setValidator(QIntValidator())
		self.textbox_geo_max.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_geo_max.addLayout(slider_layout_geo_max)
		threshold_layout_geo_max.addWidget(label_geo_max)
		threshold_layout_geo_max.addWidget(self.textbox_geo_max)
		# geometry distance from outside
		threshold_layout_geo_dist = QVBoxLayout()
		slider_layout_geo_dist = QHBoxLayout()
		self.slider_geo_dist = QSlider(Qt.Vertical)
		self.slider_geo_dist.valueChanged.connect(self.threshold_geo_dist)
		slider_layout_geo_dist.addWidget(self.slider_geo_dist)
		label_geo_dist = QLabel('distance')
		label_geo_dist.setAlignment(Qt.AlignCenter)
		self.textbox_geo_dist = QLineEdit()
		self.textbox_geo_dist.setMaxLength(4)
		self.textbox_geo_dist.setFixedWidth(50)
		self.textbox_geo_dist.setValidator(QIntValidator())
		self.textbox_geo_dist.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_geo_dist.addLayout(slider_layout_geo_dist)
		threshold_layout_geo_dist.addWidget(label_geo_dist)
		threshold_layout_geo_dist.addWidget(self.textbox_geo_dist)
		#
		threshold_layout_geo_dist_red = QVBoxLayout()
		slider_layout_geo_dist_red = QHBoxLayout()
		self.slider_geo_dist_red = QSlider(Qt.Vertical)
		self.slider_geo_dist_red.valueChanged.connect(
												self.threshold_geo_dist_red)
		slider_layout_geo_dist_red.addWidget(self.slider_geo_dist_red)
		label_geo_dist_red = QLabel('dist_red')
		label_geo_dist_red.setAlignment(Qt.AlignCenter)
		self.textbox_geo_dist_red = QLineEdit()
		self.textbox_geo_dist_red.setMaxLength(4)
		self.textbox_geo_dist_red.setFixedWidth(50)
		self.textbox_geo_dist_red.setValidator(QIntValidator())
		self.textbox_geo_dist_red.editingFinished.connect(
												self.threshold_textbox_select)
		threshold_layout_geo_dist_red.addLayout(slider_layout_geo_dist_red)
		threshold_layout_geo_dist_red.addWidget(label_geo_dist_red)
		threshold_layout_geo_dist_red.addWidget(self.textbox_geo_dist_red)
		#
		threshold_layout_geo.addLayout(threshold_layout_geo_max)
		threshold_layout_geo.addLayout(threshold_layout_geo_dist)
		threshold_layout_geo.addLayout(threshold_layout_geo_dist_red)
		tab_geo.layout.addLayout(threshold_layout_geo)
		tab_geo.setLayout(tab_geo.layout)
		tabs.addTab(tab_geo, 'geometry')
		#
		self.setup_threshold_sliders()
		self.setup_threshold_textboxes()
		options_layout.addWidget(tabs)
		zoom_layout = QVBoxLayout()
		#
		x_min_layout = QHBoxLayout()
		x_min_label = QLabel('X min:')
		x_min_label.setAlignment(Qt.AlignCenter)
		x_min_layout.addWidget(x_min_label)
		self.textbox_x_min = QLineEdit()
		self.textbox_x_min.setMaxLength(4)
		self.textbox_x_min.setFixedWidth(50)
		self.textbox_x_min.setValidator(QIntValidator())
		self.textbox_x_min.editingFinished.connect(self.bound_textbox_select)
		x_min_layout.addWidget(self.textbox_x_min)
		zoom_layout.addLayout(x_min_layout)
		#
		x_max_layout = QHBoxLayout()
		x_max_label = QLabel('X max:')
		x_max_label.setAlignment(Qt.AlignCenter)
		x_max_layout.addWidget(x_max_label)
		self.textbox_x_max = QLineEdit()
		self.textbox_x_max.setMaxLength(4)
		self.textbox_x_max.setFixedWidth(50)
		self.textbox_x_max.setValidator(QIntValidator())
		self.textbox_x_max.editingFinished.connect(self.bound_textbox_select)
		x_max_layout.addWidget(self.textbox_x_max)
		zoom_layout.addLayout(x_max_layout)
		#
		y_min_layout = QHBoxLayout()
		y_min_label = QLabel('Y min:')
		y_min_label.setAlignment(Qt.AlignCenter)
		y_min_layout.addWidget(y_min_label)
		self.textbox_y_min = QLineEdit()
		self.textbox_y_min.setMaxLength(4)
		self.textbox_y_min.setFixedWidth(50)
		self.textbox_y_min.setValidator(QIntValidator())
		self.textbox_y_min.editingFinished.connect(self.bound_textbox_select)
		y_min_layout.addWidget(self.textbox_y_min)
		zoom_layout.addLayout(y_min_layout)
		#
		y_max_layout = QHBoxLayout()
		y_max_label = QLabel('Y max:')
		y_max_label.setAlignment(Qt.AlignCenter)
		y_max_layout.addWidget(y_max_label)
		self.textbox_y_max = QLineEdit()
		self.textbox_y_max.setMaxLength(4)
		self.textbox_y_max.setFixedWidth(50)
		self.textbox_y_max.setValidator(QIntValidator())
		self.textbox_y_max.editingFinished.connect(self.bound_textbox_select)
		y_max_layout.addWidget(self.textbox_y_max)
		zoom_layout.addLayout(y_max_layout)
		#
		z_min_layout = QHBoxLayout()
		z_min_label = QLabel('Z min:')
		z_min_label.setAlignment(Qt.AlignCenter)
		z_min_layout.addWidget(z_min_label)
		self.textbox_z_min = QLineEdit()
		self.textbox_z_min.setMaxLength(4)
		self.textbox_z_min.setFixedWidth(50)
		self.textbox_z_min.setText('0')
		self.textbox_z_min.setValidator(QIntValidator())
		self.textbox_z_min.editingFinished.connect(self.bound_textbox_select)
		z_min_layout.addWidget(self.textbox_z_min)
		zoom_layout.addLayout(z_min_layout)
		#
		z_max_layout = QHBoxLayout()
		z_max_label = QLabel('Z max:')
		z_max_label.setAlignment(Qt.AlignCenter)
		z_max_layout.addWidget(z_max_label)
		self.textbox_z_max = QLineEdit()
		self.textbox_z_max.setMaxLength(4)
		self.textbox_z_max.setFixedWidth(50)
		self.textbox_z_max.setText(str(self.z_size))
		self.textbox_z_max.setValidator(QIntValidator())
		self.textbox_z_max.editingFinished.connect(self.bound_textbox_select)
		z_max_layout.addWidget(self.textbox_z_max)
		zoom_layout.addLayout(z_max_layout)
		#
		self.setup_bound_textboxes()
		#
		self.button_select = QPushButton()
		self.button_select.setText('Select Box')
		self.button_select.clicked.connect(self.select_bounds)
		zoom_layout.addWidget(self.button_select)
		#
		self.button_reset = QPushButton()
		self.button_reset.setText('Select All')
		self.button_reset.clicked.connect(self.reset_bounds)
		zoom_layout.addWidget(self.button_reset)
		#
		self.checkbox_zoom = QCheckBox("zoomed")
		self.checkbox_zoom.setChecked(self.zoomed)
		self.checkbox_zoom.stateChanged.connect(self.zoom_checkbox)
		zoom_layout.addWidget(self.checkbox_zoom)
		#
		self.checkbox_mesh = QCheckBox("plot mesh")
		self.checkbox_mesh.setChecked(self.plot_mesh)
		self.checkbox_mesh.stateChanged.connect(self.mesh_checkbox)
		zoom_layout.addWidget(self.checkbox_mesh)
		#
		self.checkbox_dapi = QCheckBox("3d dapi")
		self.checkbox_dapi.setChecked(self.plot_dapi)
		self.checkbox_dapi.stateChanged.connect(self.dapi_checkbox)
		zoom_layout.addWidget(self.checkbox_dapi)
		#
		options_layout.addLayout(zoom_layout)
		main_layout.addLayout(options_layout)
		# horizontal row of buttons
		buttons_layout = QHBoxLayout()
		#
		self.button_open_file = QPushButton()
		self.button_open_file.setText('Open Data')
		self.button_open_file.clicked.connect(self.open_file)
		buttons_layout.addWidget(self.button_open_file)
		#
		self.button_preview = QPushButton()
		self.button_preview.setText('Preview')
		self.button_preview.clicked.connect(self.preview)
		buttons_layout.addWidget(self.button_preview)
		#
		self.button_execute = QPushButton()
		self.button_execute.setText('Execute')
		self.button_execute.clicked.connect(self.execute)
		buttons_layout.addWidget(self.button_execute)
		#
		self.button_open_csv = QPushButton()
		self.button_open_csv.setText('Open CSV')
		self.button_open_csv.clicked.connect(self.open_csv)
		buttons_layout.addWidget(self.button_open_csv)
		# Layouts for advanced settings boxes
		advanced_layout = QHBoxLayout()
		neighbourhood_label = QLabel('Neighbourhood:')
		neighbourhood_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(neighbourhood_label)
		self.textbox_neighbourhood = QLineEdit()
		self.textbox_neighbourhood.setMaxLength(3)
		self.textbox_neighbourhood.setFixedWidth(40)
		self.textbox_neighbourhood.setValidator(QIntValidator())
		self.textbox_neighbourhood.setText(str(self.neighbourhood_size))
		self.textbox_neighbourhood.editingFinished.connect(
											self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_neighbourhood)
		#
		threshold_label = QLabel('Threshold Diff:')
		threshold_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(threshold_label)
		self.textbox_threshold = QLineEdit()
		self.textbox_threshold.setMaxLength(3)
		self.textbox_threshold.setFixedWidth(40)
		self.textbox_threshold.setValidator(QIntValidator())
		self.textbox_threshold.setText(str(self.threshold_difference))
		self.textbox_threshold.editingFinished.connect(
												self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_threshold)
		#
		distance_label = QLabel('Minimum Dist:')
		distance_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(distance_label)
		self.textbox_distance = QLineEdit()
		self.textbox_distance.setMaxLength(3)
		self.textbox_distance.setFixedWidth(40)
		self.textbox_distance.setValidator(QIntValidator())
		self.textbox_distance.setText(str(self.minimum_distance))
		self.textbox_distance.editingFinished.connect(
												self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_distance)
		#
		guassian_label = QLabel('Gaussian Dev:')
		guassian_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(guassian_label)
		self.textbox_guassian = QLineEdit()
		self.textbox_guassian.setMaxLength(3)
		self.textbox_guassian.setFixedWidth(40)
		self.textbox_guassian.setValidator(QIntValidator())
		self.textbox_guassian.setText(str(self.gauss_deviation))
		self.textbox_guassian.editingFinished.connect(
												self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_guassian)
		#
		layer_dist_label = QLabel('Max Layer Dist:')
		layer_dist_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(layer_dist_label)
		self.textbox_layer_distance = QLineEdit()
		self.textbox_layer_distance.setMaxLength(3)
		self.textbox_layer_distance.setFixedWidth(40)
		self.textbox_layer_distance.setValidator(QIntValidator())
		self.textbox_layer_distance.setText(str(self.max_layer_distance))
		self.textbox_layer_distance.editingFinished.connect(
												self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_layer_distance)
		#
		number_layer_label = QLabel('Min Layer Num:')
		number_layer_label.setAlignment(Qt.AlignCenter)
		advanced_layout.addWidget(number_layer_label)
		self.textbox_layer_number = QLineEdit()
		self.textbox_layer_number.setMaxLength(3)
		self.textbox_layer_number.setFixedWidth(40)
		self.textbox_layer_number.setValidator(QIntValidator())
		self.textbox_layer_number.setText(str(self.number_layer_cell))
		self.textbox_layer_number.editingFinished.connect(
												self.advanced_textbox_select)
		advanced_layout.addWidget(self.textbox_layer_number)
		#
		self.button_advanced_defaults = QPushButton()
		self.button_advanced_defaults.setText('Defaults')
		self.button_advanced_defaults.clicked.connect(self.reset_defaults)
		advanced_layout.addWidget(self.button_advanced_defaults)
		# Nest the inner layouts into the outer layout
		outer_layout.addLayout(main_layout)
		outer_layout.addLayout(buttons_layout)
		self.progress_bar = QProgressBar()
		outer_layout.addWidget(self.progress_bar)
		outer_layout.addLayout(advanced_layout)
		# Set the window's main layout
		self.setLayout(outer_layout)
	
	def setup_z_slider (self):
		self.slider_z.setMinimum(0)
		self.slider_z.setMaximum(self.z_size-1)
		self.slider_z.setSingleStep(1)
		self.slider_z.setValue(0)
	
	def setup_threshold_sliders (self):
		self.slider_green_min.setMinimum(0)
		self.slider_green_min.setMaximum(self.green_max)
		self.slider_green_min.setSingleStep(1)
		self.slider_green_min.setValue(self.green_lower)
		self.slider_green_max.setMinimum(0)
		self.slider_green_max.setMaximum(self.green_max)
		self.slider_green_max.setSingleStep(1)
		self.slider_green_max.setValue(self.green_upper)
		self.slider_green_cut.setMinimum(0)
		self.slider_green_cut.setMaximum(self.green_max)
		self.slider_green_cut.setSingleStep(1)
		self.slider_green_cut.setValue(self.green_cutoff)
		self.slider_red_min.setMinimum(0)
		self.slider_red_min.setMaximum(self.red_max)
		self.slider_red_min.setSingleStep(1)
		self.slider_red_min.setValue(self.red_lower)
		self.slider_red_max.setMinimum(0)
		self.slider_red_max.setMaximum(self.red_max)
		self.slider_red_max.setSingleStep(1)
		self.slider_red_max.setValue(self.red_upper)
		self.slider_red_cut.setMinimum(0)
		self.slider_red_cut.setMaximum(self.red_max)
		self.slider_red_cut.setSingleStep(1)
		self.slider_red_cut.setValue(self.red_cutoff)
		self.slider_geo_max.setMinimum(0)
		self.slider_geo_max.setMaximum(self.geo_size)
		self.slider_geo_max.setSingleStep(1)
		self.slider_geo_max.setValue(self.geo_edge_max)
		self.slider_geo_dist.setMinimum(0)
		self.slider_geo_dist.setMaximum(self.geo_size)
		self.slider_geo_dist.setSingleStep(1)
		self.slider_geo_dist.setValue(self.geo_distance)
		self.slider_geo_dist_red.setMinimum(0)
		self.slider_geo_dist_red.setMaximum(self.geo_size)
		self.slider_geo_dist_red.setSingleStep(1)
		self.slider_geo_dist_red.setValue(self.geo_dist_red)
	
	def setup_bound_textboxes (self):
		self.textbox_x_min.setText(str(self.x_lower))
		self.textbox_x_max.setText(str(self.x_upper))
		self.textbox_y_min.setText(str(self.y_lower))
		self.textbox_y_max.setText(str(self.y_upper))
		self.textbox_z_min.setText(str(self.z_lower))
		self.textbox_z_max.setText(str(self.z_upper))
	
	def setup_advanced_textboxes (self):
		self.textbox_neighbourhood.setText(str(self.neighbourhood_size))
		self.textbox_threshold.setText(str(self.threshold_difference))
		self.textbox_distance.setText(str(self.minimum_distance))
		self.textbox_guassian.setText(str(self.gauss_deviation))
		self.textbox_layer_distance.setText(str(self.max_layer_distance))
		self.textbox_layer_number.setText(str(self.number_layer_cell))
	
	def setup_threshold_textboxes (self):
		self.textbox_green_min.setText(str(self.green_lower))
		self.textbox_green_max.setText(str(self.green_upper))
		self.textbox_green_cut.setText(str(self.green_cutoff))
		self.textbox_red_min.setText(str(self.red_lower))
		self.textbox_red_max.setText(str(self.red_upper))
		self.textbox_red_cut.setText(str(self.red_cutoff))
		self.textbox_geo_max.setText(str(self.geo_edge_max))
		self.textbox_geo_dist.setText(str(self.geo_distance))
		self.textbox_geo_dist_red.setText(str(self.geo_dist_red))
	
	def z_min_button (self):
		self.z_lower = self.z_level
		self.setup_bound_textboxes()
	
	def z_max_button (self):
		self.z_upper = self.z_level
		self.setup_bound_textboxes()
	
	def z_textbox_select (self):
		input_z = int(self.textbox_z.text())
		if input_z > 0 and input_z < self.z_size:
			self.z_level = input_z
			self.slider_z.setValue(input_z)
			self.dapi_image, self. green_image, self.red_image = \
											self.extract_image(self.z_level)
			self.replot()
	
	def z_slider_select (self):
		self.z_level = self.slider_z.value()
		self.textbox_z.setText(str(self.z_level))
		self.dapi_image, self. green_image, self.red_image = \
											self.extract_image(self.z_level)
		self.replot()
	
	def threshold_green_lower (self):
		self.green_lower = self.slider_green_min.value()
		self.textbox_green_min.setText(str(self.green_lower))
		self.replot()
	
	def threshold_green_upper (self):
		self.green_upper = self.slider_green_max.value()
		self.textbox_green_max.setText(str(self.green_upper))
		self.replot()
	
	def threshold_green_cutoff (self):
		self.green_cutoff = self.slider_green_cut.value()
		self.textbox_green_cut.setText(str(self.green_cutoff))
		self.replot()
	
	def threshold_red_lower (self):
		self.red_lower = self.slider_red_min.value()
		self.textbox_red_min.setText(str(self.red_lower))
		self.replot()
	
	def threshold_red_upper (self):
		self.red_upper = self.slider_red_max.value()
		self.textbox_red_max.setText(str(self.red_upper))
		self.replot()
	
	def threshold_red_cutoff (self):
		self.red_cutoff = self.slider_red_cut.value()
		self.textbox_red_cut.setText(str(self.red_cutoff))
		self.replot()
	
	def threshold_geo_max (self):
		self.geo_edge_max = self.slider_geo_max.value()
		self.textbox_geo_max.setText(str(self.geo_edge_max))
		self.replot()
	
	def threshold_geo_dist (self):
		self.geo_distance = self.slider_geo_dist.value()
		self.textbox_geo_dist.setText(str(self.geo_distance))
		self.replot()
	
	def threshold_geo_dist_red (self):
		self.geo_dist_red = self.slider_geo_dist_red.value()
		self.textbox_geo_dist_red.setText(str(self.geo_dist_red))
		self.replot()
	
	def green_checkbox (self):
		self.green_active = self.checkbox_green.isChecked()
		self.replot()
	
	def red_checkbox (self):
		self.red_active = self.checkbox_red.isChecked()
		self.replot()
	
	def geo_checkbox (self):
		self.geo_active = self.checkbox_geo.isChecked()
		self.replot()
	
	def green_cutoff_checkbox (self):
		self.green_cutoff_active = self.checkbox_green_cutoff.isChecked()
		self.replot()
	
	def red_cutoff_checkbox (self):
		self.red_cutoff_active = self.checkbox_red_cutoff.isChecked()
		self.replot()
	
	def zoom_checkbox (self):
		self.zoomed = self.checkbox_zoom.isChecked()
		self.replot()
	
	def dapi_checkbox (self):
		self.plot_dapi = self.checkbox_dapi.isChecked()
		self.replot()
	
	def mesh_checkbox (self):
		self.plot_mesh = self.checkbox_mesh.isChecked()
		self.replot()
	
	def bound_textbox_select (self):
		self.x_lower = int(self.textbox_x_min.text())
		if self.x_lower < 0:
			self.x_lower = 0
		self.x_upper = int(self.textbox_x_max.text())
		if self.x_upper >= self.x_size:
			self.x_upper = self.x_size-1
		if self.x_upper < self.x_lower:
			self.x_upper = self.x_lower
		self.y_lower = int(self.textbox_y_min.text())
		if self.y_lower < 0:
			self.y_lower = 0
		self.y_upper = int(self.textbox_y_max.text())
		if self.y_upper >= self.y_size:
			self.y_upper = self.y_size-1
		if self.y_upper < self.y_lower:
			self.y_upper = self.y_lower
		self.z_lower = int(self.textbox_z_min.text())
		if self.z_lower < 0:
			self.z_lower = 0
		self.z_upper = int(self.textbox_z_max.text())
		if self.z_upper >= self.z_size:
			self.z_upper = self.z_size-1
		if self.z_upper < self.z_lower:
			self.z_upper = self.z_lower
		self.setup_bound_textboxes()
		self.replot()
	
	def threshold_textbox_select (self):
		self.green_lower = int(self.textbox_green_min.text())
		if self.green_lower < 0:
			self.green_lower = 0
		elif self.green_lower > self.green_max:
			self.green_lower = self.green_max
		self.green_upper = int(self.textbox_green_max.text())
		if self.green_upper < 0:
			self.green_upper = 0
		elif self.green_upper > self.green_max:
			self.green_upper = self.green_max
		self.green_cutoff = int(self.textbox_green_cut.text())
		if self.green_cutoff < 0:
			self.green_cutoff = 0
		elif self.green_cutoff > self.green_max:
			self.green_cutoff = self.green_max
		self.red_lower = int(self.textbox_red_min.text())
		if self.red_lower < 0:
			self.red_lower = 0
		elif self.red_lower > self.red_max:
			self.red_lower = self.red_max
		self.red_upper = int(self.textbox_red_max.text())
		if self.red_upper < 0:
			self.red_upper = 0
		elif self.red_upper > self.red_max:
			self.red_upper = self.red_max
		self.red_cutoff = int(self.textbox_red_cut.text())
		if self.red_cutoff < 0:
			self.red_cutoff = 0
		elif self.red_cutoff > self.red_max:
			self.red_cutoff = self.red_max
		self.geo_edge_max = int(self.textbox_geo_max.text())
		if self.geo_edge_max < 0:
			self.geo_edge_max = 0
		elif self.geo_edge_max > self.geo_size:
			self.geo_edge_max = self.geo_size
		self.geo_distance = int(self.textbox_geo_dist.text())
		if self.geo_distance < 0:
			self.geo_distance = 0
		elif self.geo_distance > self.geo_size:
			self.geo_distance = self.geo_size
		self.geo_dist_red = int(self.textbox_geo_dist_red.text())
		if self.geo_dist_red < 0:
			self.geo_dist_red = 0
		elif self.geo_dist_red > self.geo_size:
			self.geo_dist_red = self.geo_size
		self.setup_threshold_textboxes()
		self.setup_threshold_sliders()
	
	def advanced_textbox_select (self):
		self.neighbourhood_size = int(self.textbox_neighbourhood.text())
		self.threshold_difference = int(self.textbox_threshold.text())
		self.minimum_distance = int(self.textbox_distance.text())
		self.gauss_deviation = int(self.textbox_guassian.text())
		self.max_layer_distance = int(self.textbox_layer_distance.text())
		self.number_layer_cell = int(self.textbox_layer_number.text())
	
	def reset_defaults (self):
		self.neighbourhood_size = self.advanced_defaults[0]
		self.threshold_difference = self.advanced_defaults[1]
		self.minimum_distance = self.advanced_defaults[2]
		self.gauss_deviation = self.advanced_defaults[3]
		self.max_layer_distance = self.advanced_defaults[4]
		self.number_layer_cell = self.advanced_defaults[5]
		self.setup_advanced_textboxes()
		self.green_lower = self.threshold_defaults[0]
		self.green_upper = self.threshold_defaults[1]
		self.green_cutoff = self.threshold_defaults[2]
		self.red_lower = self.threshold_defaults[4]
		self.red_upper = self.threshold_defaults[5]
		self.red_cutoff = self.threshold_defaults[6]
		self.geo_edge_max = self.threshold_defaults[8]
		self.geo_distance = self.threshold_defaults[9]
		self.geo_dist_red = self.threshold_defaults[10]
		self.setup_threshold_textboxes()
		self.setup_threshold_sliders()
		self.checkbox_green_cutoff.setChecked(False)
		self.green_cutoff_active = False
		self.checkbox_red_cutoff.setChecked(False)
		self.red_cutoff_active = False
	
	def select_bounds (self):
		self.zoomed = False
		self.checkbox_zoom.setChecked(False)
		self.replot()
		self.clear_centres()
		self.selecting_area = True
		self.click_id = self.canvas.mpl_connect(
							'button_press_event', self.on_click)
	
	def on_click (self, event):
		if self.selecting_area:
			self.position = np.array([int(np.floor(event.xdata)),
									  int(np.floor(event.ydata))])
			self.canvas.mpl_disconnect(self.click_id)
			self.click_id = self.canvas.mpl_connect(
								'button_release_event', self.off_click)
			self.move_id = self.canvas.mpl_connect(
								'motion_notify_event', self.mouse_moved)
	
	def mouse_moved (self, event):
		if self.selecting_area:
			p_1 = np.array([int(np.floor(event.xdata)),
							int(np.floor(event.ydata))])
			p_2 = self.position
			self.canvas.plot_selector(p_1, p_2)
	
	def off_click (self, event):
		if self.selecting_area:
			p_1 = np.array([int(np.floor(event.xdata)),
							int(np.floor(event.ydata))])
			p_2 = self.position
			self.x_lower = np.amin(np.array([p_1[0], p_2[0]]))
			self.x_upper = np.amax(np.array([p_1[0], p_2[0]]))
			self.y_lower = np.amin(np.array([p_1[1], p_2[1]]))
			self.y_upper = np.amax(np.array([p_1[1], p_2[1]]))
			self.canvas.mpl_disconnect(self.click_id)
			self.canvas.mpl_disconnect(self.move_id)
			self.canvas.remove_selector()
			self.selecting_area = False
			self.setup_bound_textboxes()
			self.bound_textbox_select()
	
	def reset_bounds (self):
		self.x_lower = 0
		self.x_upper = self.x_size
		self.y_lower = 0
		self.y_upper = self.y_size
		self.setup_bound_textboxes()
		self.clear_centres()
		self.replot()
	
	def clear_centres (self):
		self.dapi_centres = np.zeros((0,2), dtype = float)
		self.green_cells = np.zeros((0,1), dtype = bool)
		self.red_cells = np.zeros((0,1), dtype = bool)
		self.epi_cells = np.zeros((0,1), dtype = bool)
		self.edges = np.zeros((0,2), dtype = int)
		self.edges_outer = np.zeros((0,1), dtype = bool)
		self.edges_outer_red = np.zeros((0,1), dtype = bool)
		self.edges_outer_green = np.zeros((0,1), dtype = bool)
	
	def replot (self):
		dapi_display = self.dapi_image
		green_display = np.where(self.green_image > self.green_lower,
							np.where(self.green_image < self.green_upper,
										self.green_image, self.green_upper), 0)
		red_display = np.where(self.red_image > self.red_lower,
							np.where(self.red_image < self.red_upper,
									self.red_image, self.red_upper), 0)
		if self.zoomed:
			self.canvas.update_images(
						dapi_display[self.y_lower:self.y_upper,
									 self.x_lower:self.x_upper],
						green_display[self.y_lower:self.y_upper,
									  self.x_lower:self.x_upper],
						red_display[self.y_lower:self.y_upper,
									self.x_lower:self.x_upper],
						show_green = self.green_active,
						show_red = self.red_active,
						box = np.array([[self.x_lower,
										 self.x_upper],
										[self.y_lower,
										 self.y_upper]]),
						show_box = False,
						show_mesh = self.plot_mesh
					)
			self.canvas.update_centres(self.dapi_centres,
									   self.green_cells,
									   self.red_cells,
									   self.epi_cells,
									   self.edges,
									   self.edges_outer,
									   self.edges_outer_red,
									   self.edges_outer_green)
		else:
			self.canvas.update_images(
						dapi_display,
						green_display,
						red_display,
						show_green = self.green_active,
						show_red = self.red_active,
						box = np.array([[self.x_lower,
										 self.x_upper],
										[self.y_lower,
										 self.y_upper]]),
						show_box = True,
						show_mesh = self.plot_mesh
					)
			self.canvas.update_centres(self.dapi_centres + \
								np.array([self.x_lower,self.y_lower]),
									   self.green_cells,
									   self.red_cells,
									   self.epi_cells,
									   self.edges,
									   self.edges_outer,
									   self.edges_outer_red,
									   self.edges_outer_green)
	
	def file_dialog (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getOpenFileName(self,
								'Open Microscope File',
								'',
								'ND2 Files (*.nd2);;' + \
								'CZI Files (*.czi);;' + \
								'IMS Files (*.ims);;' + \
								'All Files (*)',
								options=options)
		if file_name == '':
			self.file_path = None
			return False
		else:
			file_path = Path(file_name)
			if file_path.suffix.lower() == '.nd2' or \
			   file_path.suffix.lower() == '.czi' or \
			   file_path.suffix.lower() == '.ims':
				self.file_path = file_path
				return True
			else:
				self.file_path = None
				return False
	
	def open_file (self):
		if not self.file_dialog():
			return
		try:
			self.image_stack = AICSImage(str(self.file_path))
			self.channel_names = self.image_stack.channel_names
			print(self.channel_names)
			self.x_size = self.image_stack.shape[-1]
			self.y_size = self.image_stack.shape[-2]
			self.z_size = self.image_stack.shape[-3]
			self.geo_size = int(min(self.x_size,self.y_size)/8)
			self.x_lower = 0
			self.x_upper = self.x_size
			self.y_lower = 0
			self.y_upper = self.y_size
			self.z_lower = 0
			self.z_upper = self.z_size
			self.setup_bound_textboxes()
			self.scale = self.image_stack.physical_pixel_sizes[::-1]
		except:
			self.file_path = None
			display_error('Could not open file!')
			return
		self.dapi_image, self. green_image, self.red_image = \
											self.extract_image(self.z_level)
		self.setup_z_slider()
		self.setup_threshold_sliders()
		self.replot()
	
	def preview (self):
		if self.file_path == None or self.file_path == '':
			return
		if self.green_active:
			green_image = self.green_image
		else:
			green_image = None
		if self.red_active:
			red_image = self.red_image
		else:
			red_image = None
		self.dapi_centres, self.green_cells, self.red_cells = \
					self.process_image(self.dapi_image,
										green_image, red_image)
		triangulation = Delaunay(self.dapi_centres)
		self.mesh = SimplicialComplex(triangulation.points,
									  triangulation.simplices,
									  triangulation.neighbors)
		self.mesh.remove_long_simplices(self.geo_edge_max)
		self.edges, count = np.unique(np.sort(
									np.vstack([self.mesh.simplices[:,:2],
											   self.mesh.simplices[:,1:],
											   self.mesh.simplices[:,::2]]),
								axis=1), return_counts = True, axis=0)
		self.edges_outer = (count == 1)
		if self.x_lower > 0:
			self.edges_outer = self.edges_outer & \
				(self.dapi_centres[self.edges[:,0],0] > self.geo_edge_max/3) & \
				(self.dapi_centres[self.edges[:,1],0] > self.geo_edge_max/3)
		if self.y_lower > 0:
			self.edges_outer = self.edges_outer & \
				(self.dapi_centres[self.edges[:,0],1] > self.geo_edge_max/3) & \
				(self.dapi_centres[self.edges[:,1],1] > self.geo_edge_max/3)
		if self.x_upper < self.x_size-1:
			self.edges_outer = self.edges_outer & \
				(self.dapi_centres[self.edges[:,0],0] < \
					self.x_upper - self.x_lower - self.geo_edge_max/3) & \
				(self.dapi_centres[self.edges[:,1],0] < \
					self.x_upper - self.x_lower - self.geo_edge_max/3)
		if self.y_upper < self.y_size-1:
			self.edges_outer = self.edges_outer & \
				(self.dapi_centres[self.edges[:,0],1] < \
					self.y_upper - self.y_lower - self.geo_edge_max/3) & \
				(self.dapi_centres[self.edges[:,1],1] < \
					self.y_upper - self.y_lower - self.geo_edge_max/3)
		self.edges_outer_red = self.edges_outer.copy()
		outer_edges = self.edges[self.edges_outer]
		points_inner = np.ones(self.dapi_centres.shape[0], dtype = bool)
		points_inner[np.unique(outer_edges)] = False
		self.edges_outer_red[self.edges_outer] = \
					self.red_cells[outer_edges[:,0]] & \
					self.red_cells[outer_edges[:,1]]
		distances = np.zeros((self.dapi_centres.shape[0], outer_edges.shape[0]),
								dtype = float)
		for index, edge in enumerate(outer_edges):
			distances[points_inner,index] = lineseg_dists(
										self.dapi_centres[points_inner],
										self.dapi_centres[edge[0]],
										self.dapi_centres[edge[1]])
		min_indices = np.argmin(distances, axis=1)
		closest_is_red = self.edges_outer_red[self.edges_outer][min_indices]
		min_distance = np.min(distances, axis=1)
		self.epi_cells = ((min_distance < self.geo_dist_red) & \
													closest_is_red) | \
						 ((min_distance < self.geo_distance) & \
											np.logical_not(closest_is_red))
		self.replot()
	
	def execute (self):
		if self.file_path == None or self.file_path == '':
			return
		if self.z_upper <= self.z_lower:
			return
		positions_layer = np.zeros((0,3), dtype = float)
		green_cells_layer = np.zeros(0, dtype = bool)
		red_cells_layer = np.zeros(0, dtype = bool)
		self.progress_bar.setRange(self.z_lower, self.z_upper)
		self.progress_bar.setValue(self.z_lower)
		self.progress_bar.setFormat('Processing Z-Stack: %p%')
		for z_level in range(self.z_lower, self.z_upper+1):
			dapi_image, green_image, red_image = self.extract_image(z_level)
			if not self.green_active:
				green_image = None
			if not self.red_active:
				red_image = None
			dapi_centres, green_cells, red_cells = self.process_image(
											dapi_image, green_image, red_image)
			positions_layer = np.vstack([positions_layer,
				np.vstack([(dapi_centres + np.array([self.x_lower,
													 self.y_lower])).T,
						np.ones(dapi_centres.shape[0])*z_level]).T])
			green_cells_layer = np.append(green_cells_layer, green_cells)
			red_cells_layer = np.append(red_cells_layer, red_cells)
			self.progress_bar.setValue(z_level)
		self.progress_bar.reset()
		self.progress_bar.setMinimum(0)
		positions_layer_size = positions_layer.shape[0]
		self.progress_bar.setMaximum(positions_layer.shape[0])
		self.progress_bar.setValue(0)
		self.progress_bar.setFormat('Correlating Layers: %p%')
		positions = np.zeros((0,3), dtype = float)
		green_cells = np.zeros(0, dtype = bool)
		red_cells = np.zeros(0, dtype = bool)
		while positions_layer.shape[0] > 0:
			x_0, y_0, z_0 = positions_layer[0]
			found_on_next_layer = True
			layer_index = 1
			positions_temp = np.zeros((0,3), dtype = float)
			green_cells_temp = np.zeros(0, dtype = bool)
			red_cells_temp = np.zeros(0, dtype = bool)
			while found_on_next_layer and \
				  layer_index < positions_layer.shape[0]:
				found_on_next_layer = False
				# find next layer
				on_layer = (positions_layer[:,2] == z_0+1)
				layer_index = np.argmax(on_layer)
				if layer_index > 0 and \
				   layer_index < positions_layer.shape[0]:
					z_0 = positions_layer[layer_index,2]
					distances = np.linalg.norm(
										positions_layer[on_layer,:2] - \
											np.array([x_0,y_0]), axis=1)
					local_index = np.argmin(distances)
					if distances[local_index] < self.minimum_distance:
						found_on_next_layer = True
						index = layer_index + local_index
						positions_temp = np.vstack([positions_temp,
													positions_layer[index]])
						x_0 = (x_0 * positions_temp.shape[0] + \
								positions_temp[-1,0]) / \
									(positions_temp.shape[0]+1)
						y_0 = (y_0 * positions_temp.shape[0] + \
								positions_temp[-1,1]) / \
									(positions_temp.shape[0]+1)
						positions_layer = np.delete(positions_layer, index,
														axis=0)
						green_cells_temp = np.append(green_cells_temp,
													green_cells_layer[index])
						green_cells_layer = np.delete(green_cells_layer,
															index, axis=0)
						red_cells_temp = np.append(red_cells_temp,
													red_cells_layer[index])
						red_cells_layer = np.delete(red_cells_layer,
															index, axis=0)
			if positions_temp.shape[0] >= self.number_layer_cell:
				positions = np.vstack([positions, np.mean(positions_temp,
															axis=0)])
				green_cells = np.append(green_cells,
										np.count_nonzero(green_cells_temp) >= \
													green_cells_temp.shape[0]/2)
				red_cells = np.append(red_cells,
										np.count_nonzero(red_cells_temp) >= \
													red_cells_temp.shape[0]/2)
			positions_layer = np.delete(positions_layer, 0, axis=0)
			green_cells_layer = np.delete(green_cells_layer,0)
			red_cells_layer = np.delete(red_cells_layer,0)
			layers_processed = positions_layer_size - positions_layer.shape[0]
			self.progress_bar.setValue(layers_processed)
		positions = positions * self.scale
		epi_cells = np.zeros(len(red_cells), dtype = bool)
		self.progress_bar.reset()
		if self.geometry_active:
			self.progress_bar.setMinimum(0)
			self.progress_bar.setMaximum(positions.shape[0])
			self.progress_bar.setValue(0)
			self.progress_bar.setFormat('Finding Epithelial Cells: %p%')
			triangulation = Delaunay(positions)
			mesh_3d = SimplicialComplex(triangulation.points,
										triangulation.simplices,
										triangulation.neighbors)
			mesh_3d.remove_long_simplices(self.geo_edge_max)
			faces_all, count = np.unique(np.sort(
								np.vstack([mesh_3d.simplices[:,(0,1,2)],
										   mesh_3d.simplices[:,(0,1,3)],
										   mesh_3d.simplices[:,(0,2,3)],
										   mesh_3d.simplices[:,(1,2,3)]]),
								axis=1), return_counts = True, axis=0)
			faces_outer = faces_all[count == 1]
			outer_points_indices = np.unique(faces_outer)
			points = positions[outer_points_indices]
			outer_points_dict = np.zeros(positions.shape[0], dtype = int)
			outer_points_dict[outer_points_indices] = np.arange(
													len(outer_points_indices))
			faces = outer_points_dict[faces_outer]
			points_red = red_cells[outer_points_indices]
			faces_red = (points_red[faces[:,0]] & points_red[faces[:,1]]) | \
						(points_red[faces[:,0]] & points_red[faces[:,2]]) | \
						(points_red[faces[:,1]] & points_red[faces[:,2]])
			points_green = green_cells[outer_points_indices]
			faces_green = (points_green[faces[:,0]] & \
								points_green[faces[:,1]]) | \
						  (points_green[faces[:,0]] & \
								points_green[faces[:,2]]) | \
						  (points_green[faces[:,1]] & \
								points_green[faces[:,2]])
			faces_purple = faces_red & faces_green
			surface_mesh = Trimesh(vertices = points, faces = faces)
			surface_mesh.export(self.file_path.with_suffix(
					'.{0:s}.stl'.format(time.strftime("%Y.%m.%d-%H.%M.%S"))))
			fix_normals(surface_mesh)
			mask = np.zeros(faces.shape[0], dtype = bool)
			if self.x_lower > 0:
				mask = mask | \
					((points[faces[:,0],0] < (self.x_lower + \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,1],0] < (self.x_lower + \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,2],0] < (self.x_lower + \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (np.abs(surface_mesh.face_normals[:,0]) > 0.7))
			if self.y_lower > 0:
				mask = mask | \
					((points[faces[:,0],1] < (self.y_lower + \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,1],1] < (self.y_lower + \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,2],1] < (self.y_lower + \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (np.abs(surface_mesh.face_normals[:,1]) > 0.7))
			if self.z_lower > 0:
				mask = mask | \
					((points[faces[:,0],2] < (self.z_lower + \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,1],2] < (self.z_lower + \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,2],2] < (self.z_lower + \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (np.abs(surface_mesh.face_normals[:,2]) > 0.7))
			if self.x_upper < self.x_size-1:
				mask = mask | \
					((points[faces[:,0],0] > (self.x_upper - \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,1],0] > (self.x_upper - \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,2],0] > (self.x_upper - \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (np.abs(surface_mesh.face_normals[:,0]) > 0.7))
			if self.y_upper < self.y_size-1:
				mask = mask | \
					((points[faces[:,0],1] > (self.y_upper - \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,1],1] > (self.y_upper - \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,2],1] > (self.y_upper - \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (np.abs(surface_mesh.face_normals[:,1]) > 0.7))
			if self.z_upper < self.z_size-1:
				mask = mask | \
					((points[faces[:,0],2] > (self.z_upper - \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,1],2] > (self.z_upper - \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (points[faces[:,2],2] > (self.z_upper - \
									self.geo_edge_max/3) * self.scale[0]) & \
					 (np.abs(surface_mesh.face_normals[:,2]) > 0.7))
			surface_mesh.update_faces(np.logical_not(mask))
			fix_normals(surface_mesh)
			mask = np.zeros(len(surface_mesh.faces), dtype = bool)
			cc = connected_components(surface_mesh.face_adjacency, min_len=4)
			mask[np.concatenate(cc)] = True
			surface_mesh.update_faces(mask)
			#surface_mesh.show()
			#closest_points, distances, triangle_ids = \
			#		surface_mesh.nearest.on_surface(positions)
			for index, point in enumerate(positions):
				closest_point, distance, triangle_id = \
						surface_mesh.nearest.on_surface([point])
				epi_cells[index] = ((distance[0] < self.geo_dist_red * \
													self.scale[0]) and \
												faces_red[triangle_id[0]]) | \
								   ((distance[0] < self.geo_distance * \
													self.scale[0]) and \
											not faces_red[triangle_id[0]])
				self.progress_bar.setValue(index)
			###############################################################
			#face_colors = np.ones((faces.shape[0],  4), dtype = int)*150
			#face_colors[:,3] = 255
			#face_colors[faces_red] = [[200,0,0,255]]
			#face_colors[faces_green] = [[0,200,0,255]]
			#face_colors[faces_purple] = [[120,0,120,255]]
			#surface_mesh.visual.face_colors = face_colors
			#surface_mesh.show(smooth=False)
			###############################################################
			self.progress_bar.reset()
			#
		self.progress_bar.setMinimum(0)
		self.progress_bar.setFormat('')
		self.progress_bar.setMaximum(1)
		self.progress_bar.setValue(0)
		self.save_csv(positions, green_cells, red_cells, epi_cells)
		self.plot_3d(positions, green_cells, red_cells, epi_cells)
	
	def save_csv (self, positions, green_cells, red_cells, epi_cells):
		output_array = np.vstack([positions.T, green_cells, red_cells,
									epi_cells]).T
		data_format = '%.18e', '%.18e', '%.18e', '%1d', '%1d', '%1d'
		np.savetxt(self.file_path.with_suffix(
				'.{0:s}.csv'.format(time.strftime("%Y.%m.%d-%H.%M.%S"))),
				output_array, fmt = data_format, delimiter = ',',
				header = 'X,Y,Z,Is_Green,Is_Red,Is_Epithellial')
	
	def open_csv (self):
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		file_name, _ = QFileDialog.getOpenFileName(self,
								"Open CSV File",
								"",
								"CSV Files (*.csv);;All Files (*)",
								options=options)
		if file_name == '':
			return
		else:
			csv_file = Path(file_name)
		try:
			data_format = np.dtype([ ('positions', float, 3),
									 ('green_cells', bool),
									 ('red_cells', bool),
									 ('epi_cells', bool) ])
			input_data = np.loadtxt(str(csv_file), dtype = data_format,
									delimiter=',').view(np.recarray)
			self.plot_3d(input_data.positions, input_data.green_cells,
											   input_data.red_cells,
											   input_data.epi_cells)
		except:
			display_error('Could not open file!')
			return
	
	def extract_image (self, z_value):
		dapi_image = np.zeros((0,2))
		green_image = np.zeros((0,2))
		red_image = np.zeros((0,2))
		try:
			for index, channel in enumerate(self.channel_names):
				if channel == 'DAPI' or channel == 'H3258-T4':
					dapi_image = self.image_stack.get_image_data('YX',
										T = 0,
										C = index,
										Z = z_value)
				elif channel == 'Green' or channel == 'EGFP-T3':
					green_image = self.image_stack.get_image_data('YX',
										T = 0,
										C = index,
										Z = z_value)
				elif channel == 'Red' or channel == 'tdTom-T2':
					red_image = self.image_stack.get_image_data('YX',
										T = 0,
										C = index,
										Z = z_value)
		except:
			self.file_path = None
			display_error('Problem extracting data!')
		return dapi_image, green_image, red_image
	
	def process_image (self, dapi_image, green_image = None,
										red_image = None):
		dapi_centres = self.find_centres(dapi_image)
		delta = self.neighbourhood_size # int(self.neighbourhood_size/2)
		green_cells = np.zeros(dapi_centres.shape[0], dtype = bool)
		if green_image is not None:
			green_blur = green_image[self.y_lower:self.y_upper,
									 self.x_lower:self.x_upper]
			green_blur = mh.gaussian_filter(green_blur, self.gauss_deviation)
		#	green_blur = np.where(green_blur > self.green_lower,
		#					np.where(green_blur < self.green_upper,
		#								green_blur, self.green_upper), 0)
			green_blur = np.where(green_blur < self.green_upper,
										green_blur, self.green_upper)
		red_cells = np.zeros(dapi_centres.shape[0], dtype = bool)
		if red_image is not None:
			red_blur = red_image[self.y_lower:self.y_upper,
								 self.x_lower:self.x_upper]
			red_blur = mh.gaussian_filter(red_blur, self.gauss_deviation)
		#	red_blur = np.where(red_blur > self.red_lower,
		#					np.where(red_blur < self.red_upper,
		#								red_blur, self.red_upper), 0)
			red_blur = np.where(red_blur < self.red_upper,
										red_blur, self.red_upper)
		for index,(c_x,c_y) in enumerate(dapi_centres):
			# median seems to work better than mean.
			if green_image is not None:
				if np.median(green_blur[c_y-delta:c_y+delta, # mean ?
									  c_x-delta:c_x+delta]) > self.green_lower:
					green_cells[index] = True
			if red_image is not None:
				if np.median(red_blur[c_y-delta:c_y+delta, # mean ?
									c_x-delta:c_x+delta]) > self.red_lower:
					red_cells[index] = True
			if red_image is not None and green_image is not None:
				if self.green_cutoff_active and \
				   np.median(green_blur[c_y-delta:c_y+delta, # mean ?
								c_x-delta:c_x+delta]) > self.green_cutoff:
					red_cells[index] = False
				if self.red_cutoff_active and \
				   np.median(red_blur[c_y-delta:c_y+delta, # mean ?
								c_x-delta:c_x+delta]) > self.red_cutoff:
					green_cells[index] = False
		return dapi_centres, green_cells, red_cells
	
	def find_centres (self, image):
		frame = image[self.y_lower:self.y_upper,
					  self.x_lower:self.x_upper]
		frame = mh.gaussian_filter(frame, self.gauss_deviation)
		frame_max = ndi.maximum_filter(frame, self.neighbourhood_size)
		maxima = (frame == frame_max)
		frame_min = ndi.minimum_filter(frame, self.neighbourhood_size)
		differences = ((frame_max - frame_min) > self.threshold_difference)
		maxima[differences == 0] = 0
		maximum = np.amax(frame)
		minimum = np.amin(frame)
		outside_filter = (frame_max > (maximum-minimum)*0.1 + minimum)
		maxima[outside_filter == 0] = 0
		labeled, num_objects = ndi.label(maxima)
		slices = ndi.find_objects(labeled)
		centres = np.zeros((len(slices),2), dtype = int)
		good_centres = 0
		for (dy,dx) in slices:
			centres[good_centres,0] = int((dx.start + dx.stop - 1)/2)
			centres[good_centres,1] = int((dy.start + dy.stop - 1)/2)
			if centres[good_centres,0] < self.neighbourhood_size/2 or \
			   centres[good_centres,0] > (self.x_upper-self.x_lower) - \
			   								self.neighbourhood_size/2 or \
			   centres[good_centres,1] < self.neighbourhood_size/2 or \
			   centres[good_centres,1] > (self.y_upper-self.y_lower) - \
			   								self.neighbourhood_size/2:
				good_centres -= 1
			good_centres += 1
		centres = centres[:good_centres]
		return centres
	
	def plot_3d (self, positions, green_cells, red_cells, epi_cells):
		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(111, projection='3d')
		scale = 4
		if epi_cells is not None:
			if epi_cells.shape[0] > 0 and not self.green_active:
				ax.plot(positions[epi_cells,0],
						positions[epi_cells,1],
						positions[epi_cells,2],
						linestyle = '', marker = '.',
						markersize = 1.5*scale, color = 'royalblue')
			if epi_cells.shape[0] > 0 and self.green_active:
				ax.plot(positions[epi_cells & np.logical_not(green_cells),0],
						positions[epi_cells & np.logical_not(green_cells),1],
						positions[epi_cells & np.logical_not(green_cells),2],
						linestyle = '', marker = '.',
						markersize = 2.0*scale, color = 'royalblue',
						alpha = 0.5)
				ax.plot(positions[epi_cells & green_cells,0],
						positions[epi_cells & green_cells,1],
						positions[epi_cells & green_cells,2],
						linestyle = '', marker = '.',
						markersize = 2.0*scale, color = 'purple',
						alpha = 0.5)
				ax.plot(positions[np.logical_not(epi_cells) & green_cells,0],
						positions[np.logical_not(epi_cells) & green_cells,1],
						positions[np.logical_not(epi_cells) & green_cells,2],
						linestyle = '', marker = '.',
						markersize = 2.0*scale, color = 'gold',
						alpha = 0.5)
		if self.plot_dapi:
			ax.plot(positions[:,0], positions[:,1], positions[:,2],
					linestyle = '', marker = '.',
					markersize = 0.8*scale, color = 'gray',
					alpha = 0.2)
		#if self.red_active and self.green_active:
		#	ax.plot(positions[np.logical_and(red_cells,
		#						np.logical_not(green_cells)),0],
		#			positions[np.logical_and(red_cells,
		#						np.logical_not(green_cells)),1],
		#			positions[np.logical_and(red_cells,
		#						np.logical_not(green_cells)),2],
		#			linestyle = '', marker = '+',
		#			markersize = 1.0*scale, color = 'crimson')
		#	ax.plot(positions[np.logical_and(red_cells, green_cells),0],
		#			positions[np.logical_and(red_cells, green_cells),1],
		#			positions[np.logical_and(red_cells, green_cells),2],
		#			linestyle = '', marker = '+',
		#			markersize = 1.0*scale, color = 'purple')
		#	ax.plot(positions[np.logical_and(green_cells,
		#						np.logical_not(red_cells)),0],
		#			positions[np.logical_and(green_cells,
		#						np.logical_not(red_cells)),1],
		#			positions[np.logical_and(green_cells,
		#						np.logical_not(red_cells)),2],
		#				linestyle = '', marker = 'x',
		#			markersize = 0.8*scale, color = 'seagreen')
		if self.red_active and self.plot_dapi:
				ax.plot(positions[red_cells,0], positions[red_cells,1],
								positions[red_cells,2],
						linestyle = '', marker = '+',
					markersize = 0.9*scale, color = 'red')
		if self.green_active and self.plot_dapi:
				ax.plot(positions[green_cells,0], positions[green_cells,1],
								positions[green_cells,2],
						linestyle = '', marker = 'x',
					markersize = 0.7*scale, color = 'seagreen')
		ax.set_xlim([ np.amin(positions[:,0]), np.amax(positions[:,0]) ])
		ax.set_ylim([ np.amin(positions[:,1]), np.amax(positions[:,1]) ])
		ax.set_zlim([ np.amin(positions[:,2]), np.amax(positions[:,2]) ])
		ax.set_aspect('auto')
		ax.set_box_aspect((np.amax(positions[:,0]) - np.amin(positions[:,0]),
						   np.amax(positions[:,1]) - np.amin(positions[:,1]),
						   np.amax(positions[:,2]) - np.amin(positions[:,2])))
		plt.show()

################################################################################

if __name__ == "__main__":
	app = QApplication(sys.argv)
	window = Window()
	window.show()
	sys.exit(app.exec_())

################################################################################
# EOF
