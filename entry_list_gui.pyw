#!/usr/bin/python3
# -*- coding: utf-8 -*-


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
import mysql.connector
import csv
import xlsxwriter
import re


Labels = ('Name', 'Date', 'Entry Time', 'Exit Time', 'Duration', 'Count')


class Ui_MainWindow(object):
    def loadData(self):
        db = mysql.connector.connect(host="localhost", user="root", passwd="", database="amy-ai")
        sql = db.cursor()
        sql.execute("SELECT NAME, DATE, MIN(TIME) as ENTRY_TIME, MAX(TIME) as EXIT_TIME, "
                    "CONCAT(FLOOR(TIMESTAMPDIFF(MINUTE, MIN(TIME), MAX(TIME))/60), ' h ', "
                    "MOD(TIMESTAMPDIFF(MINUTE, MIN(TIME), MAX(TIME)), 60), ' m' ) AS Duration, COUNT(*) "
                    "FROM LOG WHERE DATE between '" + self.fromDate.date().toString("yyyy-MM-dd") +
                    "'  and '" + self.toDate.date().toString("yyyy-MM-dd") + "' GROUP BY DATE, NAME ")
        data = sql.fetchall()
        self.tableWidget.setRowCount(0)
        for row, row_data in enumerate(data):
            self.tableWidget.insertRow(row)
            for col, col_data in enumerate(row_data):
                col_data = col_data.decode('utf-8') if isinstance(col_data, bytes) else col_data
                self.tableWidget.setItem(row, col, QtWidgets.QTableWidgetItem(str(col_data)))
        db.close()
        if self.tableWidget.rowCount() > 0:
            self.saveasButton.setEnabled(True)
        else:
            self.saveasButton.setEnabled(False)

    def handleSave(self):
        Mfrom = self.fromDate.date().toString("MMMM")
        Mto = self.toDate.date().toString("MMMM")
        path, ext = QtWidgets.QFileDialog.getSaveFileName(self.saveasButton, 'Save File', 'Entry List of {}'
                                                          .format(Mfrom if Mfrom == Mto else Mfrom+'-'+Mto),
                                                          'CSV File(*.csv);; Excel File(*.xlsx)')
        if path and ext == "CSV File(*.csv)":
            with open(path, 'w', newline='') as stream:
                # print("saving", path)
                header = csv.DictWriter(stream, fieldnames=Labels)
                header.writeheader()
                writer = csv.writer(stream)
                for row in range(self.tableWidget.rowCount()):
                    rowdata = []
                    for column in range(self.tableWidget.columnCount()):
                        item = self.tableWidget.item(row, column)
                        if item is not None:
                            rowdata.append(item.text())
                        else:
                            rowdata.append('')
                    writer.writerow(rowdata)
                stream.close()
        elif path and ext == "Excel File(*.xlsx)":
            # writer = xlsxwriter.Workbook(path.replace('.csv', '').replace('.xlsx', '')+".xlsx")
            writer = xlsxwriter.Workbook(path)
            worksheet = writer.add_worksheet("Entry List")
            for column in range(len(Labels)):
                worksheet.write(0, column, Labels[column])

            for currentColumn in range(self.tableWidget.columnCount()):
                for currentRow in range(1, self.tableWidget.rowCount()+1):
                    try:
                        data = self.tableWidget.item(currentRow-1, currentColumn).text()
                        worksheet.write(currentRow, currentColumn, data)
                    except AttributeError:
                        pass
            writer.close()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(702, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tableWidget = QtWidgets.QTableWidget(self.centralwidget)
        self.tableWidget.setGeometry(QtCore.QRect(0, 0, 702, 430))
        self.tableWidget.setColumnCount(len(Labels))
        self.tableWidget.setHorizontalHeaderLabels(Labels)
        self.tableWidget.horizontalHeader().resizeSection(0, 150)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setRowCount(0)
        current_month = int(QtCore.QDate.currentDate().toString('MM'))
        self.previous_month = current_month-1 if current_month > 1 else 12
        self.fromDate = QtWidgets.QDateEdit(self.centralwidget)
        self.fromDate.setGeometry(QtCore.QRect(60, 450, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.fromDate.setFont(font)
        self.fromDate.setDateTime(QtCore.QDateTime(QtCore.QDate(2019, self.previous_month, 1), QtCore.QTime(0, 0, 0)))
        self.fromDate.setObjectName("fromDate")
        self.label_from = QtWidgets.QLabel(self.centralwidget)
        self.label_from.setGeometry(QtCore.QRect(10, 450, 41, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_from.setFont(font)
        self.label_from.setObjectName("label_from")
        self.toDate = QtWidgets.QDateEdit(self.centralwidget)
        self.toDate.setGeometry(QtCore.QRect(240, 450, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.toDate.setFont(font)
        self.toDate.setDateTime(QtCore.QDateTime(QtCore.QDate(2019, self.previous_month, 30), QtCore.QTime(0, 0, 0)))
        self.toDate.setObjectName("toDate")
        self.label_to = QtWidgets.QLabel(self.centralwidget)
        self.label_to.setGeometry(QtCore.QRect(190, 450, 41, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_to.setFont(font)
        self.label_to.setObjectName("label_to")
        self.loadButton = QtWidgets.QPushButton(self.centralwidget)
        self.loadButton.setGeometry(QtCore.QRect(400, 460, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.loadButton.setFont(font)
        self.loadButton.setObjectName("loadButton")

        self.loadButton.clicked.connect(self.loadData)

        self.saveasButton = QtWidgets.QPushButton(self.centralwidget)
        self.saveasButton.setGeometry(QtCore.QRect(70, 520, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.saveasButton.setFont(font)
        self.saveasButton.setObjectName("saveasButton")

        self.saveasButton.clicked.connect(self.handleSave)
        self.saveasButton.setEnabled(False)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 637, 26))
        self.menubar.setObjectName("menubar")
        self.menuX_Limited_Entry_List = QtWidgets.QMenu(self.menubar)
        self.menuX_Limited_Entry_List.setObjectName("menuX_Limited_Entry_List")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar.addAction(self.menuX_Limited_Entry_List.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Entry List"))
        MainWindow.setWindowIcon(
            QIcon(_translate("MainWindow", "C:\\Users\\shaki\\PycharmProjects\\amyFace\\x-vector.ico"))
        )
        self.label_from.setText(_translate("MainWindow", "From"))
        self.label_to.setText(_translate("MainWindow", "To"))
        self.loadButton.setText(_translate("MainWindow", "Load"))
        self.saveasButton.setText(_translate("MainWindow", "Save As"))
        self.menuX_Limited_Entry_List.setTitle(_translate("MainWindow", "X Limited Entry List"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
