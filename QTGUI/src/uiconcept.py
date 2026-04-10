# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'test.ui'
##
## Created by: Qt User Interface Compiler version 6.10.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDial, QGraphicsView,
    QGridLayout, QLabel, QMainWindow, QMenu,
    QMenuBar, QPushButton, QSizePolicy, QSpinBox,
    QStatusBar, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1359, 810)
        self.actionCreate_New = QAction(MainWindow)
        self.actionCreate_New.setObjectName(u"actionCreate_New")
        self.actionOpen_File = QAction(MainWindow)
        self.actionOpen_File.setObjectName(u"actionOpen_File")
        self.actionSave_File = QAction(MainWindow)
        self.actionSave_File.setObjectName(u"actionSave_File")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.dial = QDial(self.centralwidget)
        self.dial.setObjectName(u"dial")
        self.dial.setGeometry(QRect(10, 170, 50, 64))
        self.dial_2 = QDial(self.centralwidget)
        self.dial_2.setObjectName(u"dial_2")
        self.dial_2.setGeometry(QRect(10, 240, 50, 64))
        self.dial_3 = QDial(self.centralwidget)
        self.dial_3.setObjectName(u"dial_3")
        self.dial_3.setGeometry(QRect(10, 310, 50, 64))
        self.dial_4 = QDial(self.centralwidget)
        self.dial_4.setObjectName(u"dial_4")
        self.dial_4.setGeometry(QRect(10, 380, 50, 64))
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(200, 50, 49, 16))
        self.label.setScaledContents(False)
        self.spinBox = QSpinBox(self.centralwidget)
        self.spinBox.setObjectName(u"spinBox")
        self.spinBox.setGeometry(QRect(170, 70, 78, 26))
        self.checkBox = QCheckBox(self.centralwidget)
        self.checkBox.setObjectName(u"checkBox")
        self.checkBox.setEnabled(True)
        self.checkBox.setGeometry(QRect(780, 180, 84, 24))
        self.checkBox.setMouseTracking(True)
        self.checkBox.setTristate(False)
        self.checkBox_2 = QCheckBox(self.centralwidget)
        self.checkBox_2.setObjectName(u"checkBox_2")
        self.checkBox_2.setGeometry(QRect(780, 260, 84, 24))
        self.checkBox_3 = QCheckBox(self.centralwidget)
        self.checkBox_3.setObjectName(u"checkBox_3")
        self.checkBox_3.setGeometry(QRect(780, 330, 84, 24))
        self.checkBox_4 = QCheckBox(self.centralwidget)
        self.checkBox_4.setObjectName(u"checkBox_4")
        self.checkBox_4.setGeometry(QRect(780, 400, 84, 24))
        self.layoutWidget = QWidget(self.centralwidget)
        self.layoutWidget.setObjectName(u"layoutWidget")
        self.layoutWidget.setGeometry(QRect(80, 140, 692, 331))
        self.gridLayout = QGridLayout(self.layoutWidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.pushButton = QPushButton(self.layoutWidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton, 0, 0, 1, 1)

        self.pushButton_2 = QPushButton(self.layoutWidget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_2, 0, 1, 1, 1)

        self.pushButton_3 = QPushButton(self.layoutWidget)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_3, 0, 2, 1, 1)

        self.pushButton_5 = QPushButton(self.layoutWidget)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_5, 0, 4, 1, 1)

        self.pushButton_6 = QPushButton(self.layoutWidget)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_6, 0, 5, 1, 1)

        self.pushButton_7 = QPushButton(self.layoutWidget)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_7, 0, 6, 1, 1)

        self.pushButton_8 = QPushButton(self.layoutWidget)
        self.pushButton_8.setObjectName(u"pushButton_8")
        self.pushButton_8.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_8, 0, 7, 1, 1)

        self.pushButton_15 = QPushButton(self.layoutWidget)
        self.pushButton_15.setObjectName(u"pushButton_15")
        self.pushButton_15.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_15, 1, 0, 1, 1)

        self.pushButton_11 = QPushButton(self.layoutWidget)
        self.pushButton_11.setObjectName(u"pushButton_11")
        self.pushButton_11.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_11, 1, 1, 1, 1)

        self.pushButton_14 = QPushButton(self.layoutWidget)
        self.pushButton_14.setObjectName(u"pushButton_14")
        self.pushButton_14.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_14, 1, 2, 1, 1)

        self.pushButton_13 = QPushButton(self.layoutWidget)
        self.pushButton_13.setObjectName(u"pushButton_13")
        self.pushButton_13.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_13, 1, 3, 1, 1)

        self.pushButton_16 = QPushButton(self.layoutWidget)
        self.pushButton_16.setObjectName(u"pushButton_16")
        self.pushButton_16.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_16, 1, 4, 1, 1)

        self.pushButton_10 = QPushButton(self.layoutWidget)
        self.pushButton_10.setObjectName(u"pushButton_10")
        self.pushButton_10.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_10, 1, 5, 1, 1)

        self.pushButton_12 = QPushButton(self.layoutWidget)
        self.pushButton_12.setObjectName(u"pushButton_12")
        self.pushButton_12.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_12, 1, 6, 1, 1)

        self.pushButton_9 = QPushButton(self.layoutWidget)
        self.pushButton_9.setObjectName(u"pushButton_9")
        self.pushButton_9.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_9, 1, 7, 1, 1)

        self.pushButton_31 = QPushButton(self.layoutWidget)
        self.pushButton_31.setObjectName(u"pushButton_31")
        self.pushButton_31.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_31, 2, 0, 1, 1)

        self.pushButton_22 = QPushButton(self.layoutWidget)
        self.pushButton_22.setObjectName(u"pushButton_22")
        self.pushButton_22.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_22, 2, 1, 1, 1)

        self.pushButton_28 = QPushButton(self.layoutWidget)
        self.pushButton_28.setObjectName(u"pushButton_28")
        self.pushButton_28.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_28, 2, 2, 1, 1)

        self.pushButton_26 = QPushButton(self.layoutWidget)
        self.pushButton_26.setObjectName(u"pushButton_26")
        self.pushButton_26.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_26, 2, 3, 1, 1)

        self.pushButton_32 = QPushButton(self.layoutWidget)
        self.pushButton_32.setObjectName(u"pushButton_32")
        self.pushButton_32.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_32, 2, 4, 1, 1)

        self.pushButton_18 = QPushButton(self.layoutWidget)
        self.pushButton_18.setObjectName(u"pushButton_18")
        self.pushButton_18.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_18, 2, 5, 1, 1)

        self.pushButton_25 = QPushButton(self.layoutWidget)
        self.pushButton_25.setObjectName(u"pushButton_25")
        self.pushButton_25.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_25, 2, 6, 1, 1)

        self.pushButton_17 = QPushButton(self.layoutWidget)
        self.pushButton_17.setObjectName(u"pushButton_17")
        self.pushButton_17.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_17, 2, 7, 1, 1)

        self.pushButton_20 = QPushButton(self.layoutWidget)
        self.pushButton_20.setObjectName(u"pushButton_20")
        self.pushButton_20.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_20, 3, 0, 1, 1)

        self.pushButton_27 = QPushButton(self.layoutWidget)
        self.pushButton_27.setObjectName(u"pushButton_27")
        self.pushButton_27.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_27, 3, 1, 1, 1)

        self.pushButton_21 = QPushButton(self.layoutWidget)
        self.pushButton_21.setObjectName(u"pushButton_21")
        self.pushButton_21.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_21, 3, 2, 1, 1)

        self.pushButton_23 = QPushButton(self.layoutWidget)
        self.pushButton_23.setObjectName(u"pushButton_23")
        self.pushButton_23.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_23, 3, 3, 1, 1)

        self.pushButton_19 = QPushButton(self.layoutWidget)
        self.pushButton_19.setObjectName(u"pushButton_19")
        self.pushButton_19.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_19, 3, 4, 1, 1)

        self.pushButton_29 = QPushButton(self.layoutWidget)
        self.pushButton_29.setObjectName(u"pushButton_29")
        self.pushButton_29.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_29, 3, 5, 1, 1)

        self.pushButton_24 = QPushButton(self.layoutWidget)
        self.pushButton_24.setObjectName(u"pushButton_24")
        self.pushButton_24.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_24, 3, 6, 1, 1)

        self.pushButton_30 = QPushButton(self.layoutWidget)
        self.pushButton_30.setObjectName(u"pushButton_30")
        self.pushButton_30.setCheckable(True)
        self.pushButton_30.setAutoDefault(False)

        self.gridLayout.addWidget(self.pushButton_30, 3, 7, 1, 1)

        self.pushButton_4 = QPushButton(self.layoutWidget)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setCheckable(True)

        self.gridLayout.addWidget(self.pushButton_4, 0, 3, 1, 1)

        self.pushButton_33 = QPushButton(self.centralwidget)
        self.pushButton_33.setObjectName(u"pushButton_33")
        self.pushButton_33.setGeometry(QRect(70, 70, 81, 26))
        self.pushButton_33.setCheckable(True)
        self.graphicsView = QGraphicsView(self.centralwidget)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setGeometry(QRect(860, 80, 471, 401))
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1359, 33))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.addAction(self.actionCreate_New)
        self.menuFile.addAction(self.actionOpen_File)
        self.menuFile.addAction(self.actionSave_File)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionCreate_New.setText(QCoreApplication.translate("MainWindow", u"Create New...", None))
        self.actionOpen_File.setText(QCoreApplication.translate("MainWindow", u"Open File", None))
        self.actionSave_File.setText(QCoreApplication.translate("MainWindow", u"Save File", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"BPM", None))
        self.checkBox.setText(QCoreApplication.translate("MainWindow", u"Track 1", None))
        self.checkBox_2.setText(QCoreApplication.translate("MainWindow", u"Track 2", None))
        self.checkBox_3.setText(QCoreApplication.translate("MainWindow", u"Track 3", None))
        self.checkBox_4.setText(QCoreApplication.translate("MainWindow", u"Track 4", None))
        self.pushButton.setText("")
        self.pushButton_2.setText("")
        self.pushButton_3.setText("")
        self.pushButton_5.setText("")
        self.pushButton_6.setText("")
        self.pushButton_7.setText("")
        self.pushButton_8.setText("")
        self.pushButton_15.setText("")
        self.pushButton_11.setText("")
        self.pushButton_14.setText("")
        self.pushButton_13.setText("")
        self.pushButton_16.setText("")
        self.pushButton_10.setText("")
        self.pushButton_12.setText("")
        self.pushButton_9.setText("")
        self.pushButton_31.setText("")
        self.pushButton_22.setText("")
        self.pushButton_28.setText("")
        self.pushButton_26.setText("")
        self.pushButton_32.setText("")
        self.pushButton_18.setText("")
        self.pushButton_25.setText("")
        self.pushButton_17.setText("")
        self.pushButton_20.setText("")
        self.pushButton_27.setText("")
        self.pushButton_21.setText("")
        self.pushButton_23.setText("")
        self.pushButton_19.setText("")
        self.pushButton_29.setText("")
        self.pushButton_24.setText("")
        self.pushButton_30.setText("")
        self.pushButton_4.setText("")
        self.pushButton_33.setText(QCoreApplication.translate("MainWindow", u"Start", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
    # retranslateUi

