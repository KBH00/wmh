# gui.py

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit, QTextEdit,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, pyqtSlot, QSize

from deepwmh.main.deepwmh_seg import run_segmentation

class AspectRatioPixmapLabel(QLabel):
    def __init__(self, parent=None):
        super(AspectRatioPixmapLabel, self).__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.pixmap = None

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        if pixmap is not None:
            scaled_pixmap = self.pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super(AspectRatioPixmapLabel, self).setPixmap(scaled_pixmap)
        else:
            super(AspectRatioPixmapLabel, self).setPixmap(pixmap)

    def resizeEvent(self, event):
        if self.pixmap:
            scaled_pixmap = self.pixmap.scaled(
                self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super(AspectRatioPixmapLabel, self).setPixmap(scaled_pixmap)
        super(AspectRatioPixmapLabel, self).resizeEvent(event)

class DeepWMHGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.input_image_paths = []
        self.output_folder = ''
        self.robex_dir = ''
        self.load_settings()

    def initUI(self):
        self.setWindowTitle('DeepWMH Segmentation GUI')

        # Input Image Selection
        self.inputLabel = QLabel('Input Image(s):')
        self.inputLineEdit = QLineEdit()
        self.inputLineEdit.setReadOnly(True)
        self.inputButton = QPushButton('Browse...')
        self.inputButton.clicked.connect(self.select_input_images)

        inputLayout = QHBoxLayout()
        inputLayout.addWidget(self.inputLabel)
        inputLayout.addWidget(self.inputLineEdit)
        inputLayout.addWidget(self.inputButton)

        # Output Folder Selection
        self.outputLabel = QLabel('Output Folder:')
        self.outputLineEdit = QLineEdit()
        self.outputLineEdit.setReadOnly(True)
        self.outputButton = QPushButton('Browse...')
        self.outputButton.clicked.connect(self.select_output_folder)

        outputLayout = QHBoxLayout()
        outputLayout.addWidget(self.outputLabel)
        outputLayout.addWidget(self.outputLineEdit)
        outputLayout.addWidget(self.outputButton)

        # ROBEX Directory Selection
        self.robexLabel = QLabel('ROBEX Directory:')
        self.robexLineEdit = QLineEdit()
        self.robexLineEdit.setReadOnly(True)
        self.robexButton = QPushButton('Browse...')
        self.robexButton.clicked.connect(self.select_robex_dir)

        robexLayout = QHBoxLayout()
        robexLayout.addWidget(self.robexLabel)
        robexLayout.addWidget(self.robexLineEdit)
        robexLayout.addWidget(self.robexButton)

        # Execute Button
        self.executeButton = QPushButton('Execute Segmentation')
        self.executeButton.clicked.connect(self.start_segmentation)

        # Log Text Area
        self.logLabel = QLabel('Logs:')
        self.logTextEdit = QTextEdit()
        self.logTextEdit.setReadOnly(True)

        # Volume Display
        self.volumeLabel = QLabel('Calculated Volumes:')
        self.volumeTextEdit = QTextEdit()
        self.volumeTextEdit.setReadOnly(True)

        # GIF Display
        self.gifLabel = QLabel('Result GIF:')
        self.gifDisplay = AspectRatioPixmapLabel()
        self.gifDisplay.setAlignment(Qt.AlignCenter)

        # Set size policy to allow expansion
        self.gifDisplay.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Layout Setup
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(inputLayout)
        mainLayout.addLayout(outputLayout)
        mainLayout.addLayout(robexLayout)
        mainLayout.addWidget(self.executeButton)
        mainLayout.addWidget(self.logLabel)
        mainLayout.addWidget(self.logTextEdit)
        mainLayout.addWidget(self.volumeLabel)
        mainLayout.addWidget(self.volumeTextEdit)
        mainLayout.addWidget(self.gifLabel)
        mainLayout.addWidget(self.gifDisplay)  # Add gifDisplay directly

        self.setLayout(mainLayout)
        self.resize(800, 800)

    def select_input_images(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            'Select Input Image(s)',
            '',
            'NIfTI Files (*.nii *.nii.gz)',
            options=options
        )
        if files:
            self.inputLineEdit.setText('; '.join(files))
            self.input_image_paths = files
            self.save_settings()

    def select_output_folder(self):
        options = QFileDialog.Options()
        folder = QFileDialog.getExistingDirectory(
            self,
            'Select Output Folder',
            '',
            options=options
        )
        if folder:
            self.outputLineEdit.setText(folder)
            self.output_folder = folder
            self.save_settings()

    def select_robex_dir(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(
            self,
            'Select ROBEX Directory',
            '',
            options=options
        )
        if directory:
            self.robexLineEdit.setText(directory)
            self.robex_dir = directory
            self.save_settings()

    def load_settings(self):
        settings = QSettings('YourOrganization', 'DeepWMHGUI')
        self.robex_dir = settings.value('ROBEX_DIR', '', type=str)
        self.robexLineEdit.setText(self.robex_dir)

        self.input_image_paths = settings.value('INPUT_IMAGES', [], type=list)
        self.output_folder = settings.value('OUTPUT_FOLDER', '', type=str)

        self.inputLineEdit.setText('; '.join(self.input_image_paths))
        self.outputLineEdit.setText(self.output_folder)

    def save_settings(self):
        settings = QSettings('YourOrganization', 'DeepWMHGUI')
        settings.setValue('ROBEX_DIR', self.robex_dir)
        settings.setValue('INPUT_IMAGES', self.input_image_paths)
        settings.setValue('OUTPUT_FOLDER', self.output_folder)

    def start_segmentation(self):
        if not self.input_image_paths:
            QMessageBox.warning(self, 'Input Error', 'Please select at least one input image.')
            return

        if not self.output_folder:
            QMessageBox.warning(self, 'Input Error', 'Please select an output folder.')
            return

        if not self.robex_dir:
            QMessageBox.warning(self, 'Input Error', 'Please select the ROBEX directory.')
            return

        # Disable the execute button to prevent multiple clicks
        self.executeButton.setEnabled(False)
        self.logTextEdit.clear()
        self.volumeTextEdit.clear()
        self.gifDisplay.clear()

        # Start the segmentation in a separate thread
        self.segmentationThread = SegmentationThread(
            self.input_image_paths, self.output_folder, self.robex_dir)
        self.segmentationThread.log_signal.connect(self.update_log, Qt.QueuedConnection)
        self.segmentationThread.volume_signal.connect(self.update_volume, Qt.QueuedConnection)
        self.segmentationThread.gif_signal.connect(self.update_gif, Qt.QueuedConnection)
        self.segmentationThread.finished_signal.connect(self.segmentation_finished, Qt.QueuedConnection)
        self.segmentationThread.start()

    @pyqtSlot(str)
    def update_log(self, text):
        self.logTextEdit.append(text)

    @pyqtSlot(str)
    def update_volume(self, text):
        self.volumeTextEdit.append(text)

    @pyqtSlot(str)
    def update_gif(self, gif_path):
        print(f'update_gif called with gif_path: {gif_path}')
        if os.path.exists(gif_path):
            try:
                self.movie = QMovie(gif_path)
                if not self.movie.isValid():
                    self.logTextEdit.append(f'Invalid GIF file: {gif_path}')
                    return

                # Connect the frameChanged signal to update the pixmap
                self.movie.frameChanged.connect(self.on_frame_changed)

                self.movie.start()

                self.logTextEdit.append(f'Displaying GIF: {gif_path}')
            except Exception as e:
                self.logTextEdit.append(f'Error loading GIF: {e}')
        else:
            print(f'GIF not found at {gif_path}')
            self.logTextEdit.append(f'GIF not found at {gif_path}')

    def on_frame_changed(self, frame_number):
        # Get the current frame as a pixmap
        pixmap = self.movie.currentPixmap()
        # Set it to the custom label, which handles aspect ratio
        self.gifDisplay.setPixmap(pixmap)

    def segmentation_finished(self):
        self.executeButton.setEnabled(True)
        QMessageBox.information(self, 'Segmentation Completed', 'The segmentation process has finished.')

class SegmentationThread(QThread):
    log_signal = pyqtSignal(str)
    volume_signal = pyqtSignal(str)
    gif_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, input_image_paths, output_folder, robex_dir):
        super().__init__()
        self.input_image_paths = input_image_paths
        self.output_folder = output_folder
        self.robex_dir = robex_dir

    def run(self):
        # Redirect stdout and stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = EmittingStream(text_written=self.log_signal)
        sys.stderr = EmittingStream(text_written=self.log_signal)

        try:
            # Set the ROBEX_DIR environment variable
            os.environ['ROBEX_DIR'] = self.robex_dir

            for input_image in self.input_image_paths:
                run_segmentation(input_image, self.output_folder, self)
        except Exception as e:
            import traceback
            error_message = ''.join(traceback.format_exception(None, e, e.__traceback__))
            self.log_signal.emit(f'Error: {error_message}')

        # Restore stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        self.finished_signal.emit()

class EmittingStream:
    def __init__(self, text_written):
        self.text_written = text_written

    def write(self, text):
        if text.strip() != '':
            self.text_written.emit(text)

    def flush(self):
        pass

def main():
    app = QApplication(sys.argv)
    gui = DeepWMHGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
