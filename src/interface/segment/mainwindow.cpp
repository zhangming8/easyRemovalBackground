#include <QFileDialog>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QPainter>
#include <QRandomGenerator>
#include <QTime>
#include <QElapsedTimer>
#include <QDebug>
#include <QMessageBox>
#include <QImageReader>
#include <QActionGroup>
#include <QSysInfo>

#include "interface/segment/mainwindow.h"

MainWindow::MainWindow(QWidget *parent, Args &args)
    : QMainWindow(parent), args_(args), current_model_index_(0), device_id_(0)
{
    logger_ = Logger::GetInstance(args_.root_path, "easyRemovalBackground.log", "removeBg").get_logger();
    loadConfigWithQSettings();
    initUI();
    // initModels();
    setAcceptDrops(true);
}

MainWindow::~MainWindow()
{
}

void MainWindow::initUI()
{
    setWindowTitle(QString("%1 %2").arg(QString::fromStdString(args_.app_name)).arg(QString::fromStdString(args_.app_version)));
    setMinimumSize(win_w_, win_h_);

    // 创建菜单栏
    QMenuBar *menu_bar = new QMenuBar(this);
    setMenuBar(menu_bar);

    // 文件菜单
    QMenu *file_menu = menu_bar->addMenu("File");
    QAction *open_action = new QAction("Load Image", this);
    connect(open_action, &QAction::triggered, this, &MainWindow::loadImage);
    file_menu->addAction(open_action);
    QAction *save_action = new QAction("Save Result", this);
    connect(save_action, &QAction::triggered, this, &MainWindow::saveResults);
    file_menu->addAction(save_action);

    // 模型菜单
    QMenu *model_menu = menu_bar->addMenu("Model");
    QActionGroup *model_action_group = new QActionGroup(this);
    model_action_group->setExclusive(true); // 设置为单选模式
    for (int i = 0; i < model_list_names_.size(); ++i)
    {
        QAction *action = new QAction(model_list_names_[i], this);
        action->setCheckable(true); // 设置为可勾选
        action->setData(i);         // 存储索引

        if (i == current_model_index_)
            action->setChecked(true);
        model_action_group->addAction(action);
        model_menu->addAction(action);
    }
    connect(model_action_group, &QActionGroup::triggered, this, &MainWindow::onModelSelected);

    // 设备菜单
    QMenu *device_menu = menu_bar->addMenu("Device");
    // 创建 ActionGroup 确保单选行为
    QActionGroup *device_action_group = new QActionGroup(this);
    device_action_group->setExclusive(true); // 设置为单选模式

    // 添加设备选项
    QStringList current_devices = {};
    std::vector<std::string> cuda_devices = get_devices();
    for (int i = 0; i < cuda_devices.size(); ++i)
    {
        std::string device = cuda_devices.at(i) + "(" + std::to_string(i) + ")";
        qDebug() << "Found device:" << QString::fromStdString(device);
        current_devices.append(QString::fromStdString(device));
        logger_->info("Found device {}/{}: {}", i, cuda_devices.size(), device);
    }
    current_devices.append("CPU");
    if (current_devices == devices_)
    {
        for (int i = 0; i < devices_.size(); ++i)
        {
            QAction *action = new QAction(devices_[i], this);
            action->setCheckable(true); // 设置为可勾选
            action->setData(i);         // 存储索引

            if (i == device_id_ || (devices_[i] == "CPU" && device_id_ == -1))
            {
                action->setChecked(true);
            }
            device_action_group->addAction(action);
            device_menu->addAction(action);
        }
        qDebug() << "Device list unchanged.";
    }
    else
    {
        devices_ = current_devices;
        for (int i = 0; i < devices_.size(); ++i)
        {
            QAction *action = new QAction(devices_[i], this);
            action->setCheckable(true); // 设置为可勾选
            action->setData(i);         // 存储索引

            // 设置默认选中第一个
            if (i == 0)
            {
                action->setChecked(true);
                device_id_ = i;
                if (devices_[i] == "CPU")
                {
                    device_id_ = -1;
                }
            }
            device_action_group->addAction(action);
            device_menu->addAction(action);
        }
        qDebug() << "Updated device list:" << devices_;
    }

    connect(device_action_group, &QActionGroup::triggered, this, &MainWindow::onDeviceSelected);

    // 帮助菜单
    QMenu *help_menu = menu_bar->addMenu("Help");
    QAction *about_action = new QAction("About", this);
    connect(about_action, &QAction::triggered, this, &MainWindow::showAboutDialog);
    help_menu->addAction(about_action);

    // 中央部件
    QWidget *centra_widget = new QWidget(this);
    setCentralWidget(centra_widget);

    // 主布局
    QHBoxLayout *main_layout = new QHBoxLayout(centra_widget);
    main_layout->setSpacing(30);
    main_layout->setContentsMargins(10, 10, 10, 10);

    // 左侧布局 - 原始图片显示区
    QVBoxLayout *left_layout = new QVBoxLayout();
    main_layout->addLayout(left_layout, 1);

    // 原始图片显示标签
    image_label_ = new QLabel(this);
    image_label_->setAlignment(Qt::AlignCenter);
    image_label_->setStyleSheet("background-color: white; border: 1px solid #cccccc; min-height: 400px;");
    image_label_->setAcceptDrops(true);
    image_label_->setText("Load or drag image or double click here");
    left_layout->addWidget(image_label_);

    // 加载图片按钮
    load_button_ = new QPushButton("Load Image", this);
    connect(load_button_, &QPushButton::clicked, this, &MainWindow::loadImage);
    left_layout->addWidget(load_button_);

    // 右侧布局 - 分割结果显示区
    QVBoxLayout *right_layout = new QVBoxLayout();
    main_layout->addLayout(right_layout, 1);

    // 分割结果显示标签
    result_label_ = new QLabel(this);
    result_label_->setAlignment(Qt::AlignCenter);
    result_label_->setStyleSheet("background-color: white; border: 1px solid #cccccc; min-height: 400px;");
    result_label_->setText("Result will be displayed here");
    right_layout->addWidget(result_label_);

    // 保存结果按钮
    save_button_ = new QPushButton("Save Result", this);
    save_button_->setEnabled(false);
    connect(save_button_, &QPushButton::clicked, this, &MainWindow::saveResults);
    right_layout->addWidget(save_button_);

    // 设置整体样式
    setStyleSheet("QMainWindow { background-color: #f0f0f0; }"
                  "QPushButton { background-color: #4a90e2; color: white; border: none; padding: 8px 16px; border-radius: 4px; }"
                  "QPushButton:hover { background-color: #357abd; }"
                  "QPushButton:disabled { background-color: #cccccc; }"
                  "QComboBox { border: 1px solid #cccccc; padding: 4px; border-radius: 4px; }"
                  "QLabel { color: #333333; font-size: 15px; }"
                  "QListWidget { font-size: 14px; }");
    saveConfigWithQSettings();
}

void MainWindow::initModels()
{
    // 初始化分割模型
    // std::string weight = args_.root_path + "/weights/BiRefNet_dynamic-general-epoch_174.engine";
    // std::string weight = args_.root_path + "/weights/BiRefNet_dynamic-general-epoch_174.onnx";
    // std::string weight = args_.root_path + "/weights/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx";
    std::string weight = args_.root_path + "/" + model_list_path_.at(current_model_index_).toStdString();

    if (!QFile::exists(QString::fromStdString(weight)))
    {
        QMessageBox::critical(this, "model not found", QString("model file not found: %1").arg(QString::fromStdString(weight)));
        exit(-1);
    }
    logger_->info("Loading model: {}", weight);
    logger_->info("device id: {} model name: '{}'", device_id_, model_list_names_.at(current_model_index_).toStdString());

    matting_model_ = std::make_shared<matting::Matter>(weight, logger_, device_id_);
}

void MainWindow::doSegmentation(QString file_path)
{
    if (!file_path.isEmpty())
    {
        QMessageLogger().debug() << "Selected file:" << file_path;
        QImageReader reader(file_path);
        if (!reader.canRead())
        {
            qDebug() << "QImageReader cannot read the image: " << reader.errorString();
            qDebug() << "Supported formats:" << QImageReader::supportedImageFormats();
            QMessageBox::warning(this, "read failed", QString("cannot open file: %1, supported format %2").arg(file_path, QString(reader.supportedImageFormats().join(", "))));
            return;
        }

        // // 设置期望的格式
        reader.setFormat("JPEG"); // 或 "PNG" 等
        reader.setAutoTransform(true);

        original_image_ = reader.read();
        if (!original_image_.isNull())
        {
            current_image_path_ = file_path;
            original_image_ = original_image_.convertToFormat(QImage::Format_RGB888);

            original_pixmap_ = QPixmap::fromImage(original_image_);
            // QMessageBox::warning(this, "read success",  QString("loaded file: %1 (%2,%3)").arg(file_path).arg(original_pixmap_.width()).arg(original_pixmap_.height()));

            // 调整图片大小以适应显示区域
            QSize label_size = image_label_->size();
            QPixmap scaled_pixmap = original_pixmap_.scaled(label_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            if (matting_model_ == nullptr)
            {
                // QMessageBox::information(this, "waitting", QString("waiting for model loading in first running, type '") + model_list_names_.at(current_model_index_) + QString("'"));
                // image_label_->setStyleSheet("font-size: 20px; color: red;");
                image_label_->setStyleSheet(
                    "QLabel {"
                    "font-size: 20px;"
                    "color: #d35400;"
                    "font-weight: bold;"
                    "background-color: #fef9e7;"
                    "border: 2px dashed #f39c12;"
                    "border-radius: 5px;"
                    "padding: 10px;"
                    "}");
                image_label_->setText(QString::fromStdString("Initializing model '%1'...\nThis may take a moment on first run.").arg(model_list_names_.at(current_model_index_))); // 模型第一次运行时初始化的提示信息
                QCoreApplication::processEvents();                                                                                                                                 // 强制UI立即更新
                initModels();
                image_label_->setStyleSheet("background-color: white; border: 1px solid #cccccc; min-height: 400px;");
            }
            image_label_->setPixmap(scaled_pixmap);
            image_label_->setText(""); // 清除提示文本
            QMessageLogger().debug() << "Image loaded with size:" << original_pixmap_.size();

            // 清空之前的分割结果
            result_label_->setText("segmentation result");
            save_button_->setEnabled(false);

            // 自动运行分割
            runSegmentation();
        }
        else
        {
            QMessageBox::warning(this, "read failed", QString("cannot open file: %1").arg(file_path));
        }
    }
}

void MainWindow::loadImage()
{
    // 打开文件对话框
    QString file_path = QFileDialog::getOpenFileName(this, "select image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.wbmp *.webp);;All Files (*)");
    doSegmentation(file_path);
}

cv::Mat MainWindow::QImageToCvMat(const QImage &image, bool clone_data)
{
    if (image.isNull())
    {
        return cv::Mat();
    }

    cv::Mat mat;

    switch (image.format())
    {
    case QImage::Format_Indexed8:
    case QImage::Format_Grayscale8:
    {
        QMessageLogger().debug() << "QImage::Format_Indexed8 or QImage::Format_Grayscale8";
        cv::Mat mat = cv::Mat(image.height(), image.width(), CV_8UC1, (void *)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_GRAY2RGB);
        if (clone_data)
            mat = mat.clone();
        break;
    }
    case QImage::Format_RGB888:
    {
        QMessageLogger().debug() << "QImage::Format_RGB888";
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, (void *)image.constBits(), image.bytesPerLine());
        // cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        if (clone_data)
            mat = mat.clone();
        break;
    }
    case QImage::Format_RGB32:
    case QImage::Format_ARGB32:
    case QImage::Format_ARGB32_Premultiplied:
    {
        QMessageLogger().debug() << "QImage::Format_ARGB32 or QImage::Format_ARGB32_Premultiplied";
        mat = cv::Mat(image.height(), image.width(), CV_8UC4, (void *)image.constBits(), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_RGBA2RGB);
        if (clone_data)
            mat = mat.clone();
        break;
    }
    default:
    {
        // 转换为RGB888格式
        QMessageLogger().debug() << "QImage::Other Format, converting to RGB888";
        QImage converted = image.convertToFormat(QImage::Format_RGB888);
        mat = cv::Mat(converted.height(), converted.width(), CV_8UC3, (void *)converted.constBits(), converted.bytesPerLine());
        // cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
        if (clone_data)
            mat = mat.clone();
        break;
    }
    }

    return mat;
}

QImage MainWindow::cvMatToQImage(const cv::Mat &mat)
{
    switch (mat.type())
    {
    case CV_8UC4:
    {
        QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_ARGB32);
        return image.copy();
    }
    case CV_8UC3:
    {
        QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_RGB888);
        // image = image.rgbSwapped();
        return image.copy();
    }
    case CV_8UC1:
    {
        QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8);
        return image.copy();
    }
    default:
        cv::Mat converted;
        cv::cvtColor(mat, converted, cv::COLOR_BGR2RGB);
        QImage image(converted.data, converted.cols, converted.rows, static_cast<int>(converted.step), QImage::Format_RGB888);
        return image.copy();
    }
}

void MainWindow::runSegmentation()
{
    QMessageLogger().debug() << "Running segmentation...";
    if (original_pixmap_.isNull())
    {
        return;
    }

    // 记录分割开始时间
    QElapsedTimer timer;
    timer.start();

    int image_width = original_image_.width();
    int image_height = original_image_.height();

    cv::Mat rgb_image = QImageToCvMat(original_image_, true), bgr_image;
    // cv::cvtColor(rgb_image, bgr_image, cv::COLOR_RGB2BGR);
    // cv::imwrite(args_.root_path + "/debug_input.jpg", bgr_image);

    ObjectMatting matting_result = matting_model_->run(rgb_image);
    // cv::imwrite(args_.root_path + "/debug_output.png", matting_result.alpha);

    cv::Mat alpha2, fg = rgb_image.clone(), bg = rgb_image.clone(), blended;
    bg.setTo(cv::Scalar(255, 255, 255));
    cv::cvtColor(matting_result.alpha, alpha2, cv::COLOR_GRAY2RGB);

    cv::multiply(fg, alpha2, fg, 1.0 / 255.0);                             // fg * alpha / 255
    cv::multiply(bg, cv::Scalar(255, 255, 255) - alpha2, bg, 1.0 / 255.0); // bg * (255 - alpha) / 255
    cv::add(fg, bg, blended);                                              // fg * alpha / 255 + bg * (255 - alpha) / 255
    // cv::imwrite(args_.root_path + "/debug_blended.jpg", blended);

    // QImage result = cvMatToQImage(matting_result.alpha);
    QImage result = cvMatToQImage(blended);
    // result.save(std::string(args_.root_path + "/debug_result.png").c_str());

    result_pixmap_ = QPixmap::fromImage(result);

    // 显示带有分割结果的图片
    QSize label_size = result_label_->size();
    QPixmap scaled_pixmap = result_pixmap_.scaled(label_size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    result_label_->setPixmap(scaled_pixmap);

    // 启用保存按钮
    save_button_->setEnabled(true);
}

void MainWindow::saveResults()
{
    if (result_pixmap_.isNull())
    {
        QMessageBox::warning(this, "Save fialed", "No result to save!");
        return;
    }

    if (current_image_path_.isEmpty())
    {
        QMessageBox::warning(this, "Save fialed", "No image loaded!");
        return;
    }

    QFileInfo file_info(current_image_path_);
    // QString fileNameWithoutExtension = file_info.completeBaseName();

    // 只去掉后缀
    QString path_without_ext = file_info.path() + "/" + file_info.completeBaseName();

    QString file_path = QFileDialog::getSaveFileName(this, "Save result", path_without_ext + "_result", "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;BMP Images (*.bmp);;All Files (*)");

    if (!file_path.isEmpty())
    {
        // 保存图片
        if (result_pixmap_.save(file_path))
        {
            QMessageBox::information(this, "Save success", "Save to " + file_path);
        }
        else
        {
            QMessageBox::warning(this, "Save fialed", "Cannot save segmentation result!");
        }
    }
}

void MainWindow::dragEnterEvent(QDragEnterEvent *event)
{
    QMessageLogger().debug() << "dragEnterEvent...";
    if (event->mimeData()->hasUrls() && event->mimeData()->urls().size() == 1)
    {
        event->acceptProposedAction();
    }
}

void MainWindow::dropEvent(QDropEvent *event)
{
    QMessageLogger().debug() << "dropEvent...";
    if (event->mimeData()->hasUrls() && event->mimeData()->urls().size() == 1)
    {
        QUrl url = event->mimeData()->urls().first();
        if (url.isLocalFile())
        {
            QString file_path = url.toLocalFile();
            doSegmentation(file_path);
        }
    }
}

void MainWindow::mouseDoubleClickEvent(QMouseEvent *event)
{
    QMessageLogger().debug() << "mouseDoubleClickEvent...";
    if (event->button() == Qt::LeftButton)
    {
        loadImage();

        event->accept();
    }
    else
    {
        event->ignore();
    }
}

void MainWindow::onDeviceSelected(QAction *action)
{

    int index = action->data().toInt();
    QString device_name = action->text();
    bool is_checked = action->isChecked();

    qDebug() << "Selected device:" << device_name << "Index:" << index << "Checked:" << is_checked;
    if (is_checked)
    {
        if (device_name == "CPU")
            index = -1;
        if (device_id_ != index)
        {
            device_id_ = index;
            logger_->info("Switched to device: '{}' device_id: {}", device_name.toStdString(), device_id_);

            initModels();
            saveConfigWithQSettings();
            QMessageBox::information(this, "Device switched", QString("Switched to device: %1").arg(device_name));
        }
    }
}

void MainWindow::onModelSelected(QAction *action)
{

    int index = action->data().toInt();
    QString model_name = action->text();
    bool is_checked = action->isChecked();

    qDebug() << "Selected model:" << model_name << "Index:" << index << "Checked:" << is_checked;
    if (is_checked)
    {
        if (current_model_index_ != index)
        {
            current_model_index_ = index;
            logger_->info("Switched to model: '{}' model_index: {}", model_name.toStdString(), current_model_index_);

            initModels();
            saveConfigWithQSettings();
            QMessageBox::information(this, "Model switched", QString("Switched to model: %1").arg(model_name));
        }
    }
}

void MainWindow::saveConfigWithQSettings()
{
    QSettings settings(config_file_path_, QSettings::IniFormat);

    settings.setValue("application/name", QString::fromStdString(args_.app_name));
    settings.setValue("application/version", QString::fromStdString(args_.app_version));
    settings.setValue("application/email", QString::fromStdString(args_.email));
    settings.setValue("application/website", QString::fromStdString(args_.website));

    settings.setValue("window/width", win_w_);
    settings.setValue("window/height", win_h_);

    settings.setValue("model/names", model_list_names_);
    settings.setValue("model/path", model_list_path_);
    settings.setValue("model/index", current_model_index_);

    settings.setValue("device/names", devices_);
    settings.setValue("device/index", device_id_);

    settings.sync();

    logger_->info("Config saved to: {}", config_file_path_.toStdString());
}

void MainWindow::loadConfigWithQSettings()
{
    logger_->info("---------------------------------");
    // logger_->info("Machine host name: {}", QSysInfo::machineHostName().toStdString());
    logger_->info("System: {}", QSysInfo::prettyProductName().toStdString());
    logger_->info("Product type: {}", QSysInfo::productType().toStdString());
    logger_->info("Product version: {}", QSysInfo::productVersion().toStdString());
    logger_->info("Kernel type: {}", QSysInfo::kernelType().toStdString());
    logger_->info("Kernel version: {}", QSysInfo::kernelVersion().toStdString());
    logger_->info("Current cpu architecture: {}", QSysInfo::currentCpuArchitecture().toStdString());
    logger_->info("Build cpu architecture: {}", QSysInfo::buildCpuArchitecture().toStdString());

    MEMORYSTATUSEX memoryStatus;
    memoryStatus.dwLength = sizeof(memoryStatus);
    GlobalMemoryStatusEx(&memoryStatus);
    logger_->info("Total physical memory: {} MB", memoryStatus.ullTotalPhys / (1024 * 1024));
    logger_->info("Available physical memory: {} MB", memoryStatus.ullAvailPhys / (1024 * 1024));
    logger_->info("Memory usage: {}%", memoryStatus.dwMemoryLoad);

    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    logger_->info("Processor count: {}", sysinfo.dwNumberOfProcessors);
    logger_->info("Processor level: {}", sysinfo.wProcessorLevel);
    logger_->info("Processor architecture: {}", sysinfo.wProcessorArchitecture);

    logger_->info("Loading config from: {}", config_file_path_.toStdString());
    QSettings settings(config_file_path_, QSettings::IniFormat);

    win_w_ = settings.value("window/width", win_w_).toInt();
    win_h_ = settings.value("window/height", win_h_).toInt();

    model_list_names_ = settings.value("model/names", model_list_names_).toStringList();
    model_list_path_ = settings.value("model/path", model_list_path_).toStringList();
    current_model_index_ = settings.value("model/index", 0).toInt();

    devices_ = settings.value("device/names", devices_).toStringList();
    device_id_ = settings.value("device/index", -1).toInt();

    logger_->info("Config loaded: model index {}, device id {}", current_model_index_, device_id_);
    logger_->info("Model names: {}", model_list_names_.join(", ").toStdString());
    logger_->info("Model paths: {}", model_list_path_.join(", ").toStdString());
}

void MainWindow::showAboutDialog()
{
    AboutDialog dialog(this, args_);
    dialog.exec();
}
