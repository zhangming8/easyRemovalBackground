#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QListWidget>
#include <QComboBox>
#include <QPixmap>
#include <QImage>
#include <QPoint>
#include <QRect>
#include <QVector>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QMap>
#include <QSettings>
#include <QTextEdit>
#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QClipboard>
#include <QGuiApplication>

#include <opencv2/opencv.hpp>

#include "app.h"

class AboutDialog : public QDialog
{
    Q_OBJECT

public:
    Args args_;

    explicit AboutDialog(QWidget *parent = nullptr, Args &args = Args{}) : QDialog(parent), args_(args)
    {
        setupUI();
    }

private:
    void setupUI()
    {
        setWindowTitle(tr("About"));
        setMinimumSize(450, 300);

        QVBoxLayout *main_layout = new QVBoxLayout(this);

        // QLabel *title_label = new QLabel(tr("<h2>%1</h2>").arg(app_name_));
        // title_label->setAlignment(Qt::AlignCenter);

        // 使用 QTextEdit 替代 QLabel，支持复制
        QTextEdit *info_text = new QTextEdit();
        info_text->setHtml(getVersionInfo());
        info_text->setReadOnly(true); // 只读，但可以选择和复制
        info_text->setFrameStyle(QFrame::NoFrame);
        info_text->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

        // 复制按钮
        QPushButton *copy_buttom = new QPushButton(tr("Copy"));
        connect(copy_buttom, &QPushButton::clicked, this, &AboutDialog::copyToClipboard);

        // 确定按钮
        QPushButton *ok_button = new QPushButton(tr("OK"));
        connect(ok_button, &QPushButton::clicked, this, &AboutDialog::accept);

        // 按钮布局
        QHBoxLayout *button_layout = new QHBoxLayout();
        button_layout->addWidget(copy_buttom);
        button_layout->addStretch();
        button_layout->addWidget(ok_button);

        // main_layout->addWidget(title_label);
        main_layout->addWidget(info_text);
        main_layout->addLayout(button_layout);
    }

    QString getVersionInfo()
    {
        return QString(
                   "<p><b>Application:</b> %1</p>"
                   "<p><b>Version:</b> %2</p>"
                   "<p><b>Build Date:</b> %3 %4</p>"
                   "<p><b>Contact:</b> %5</p>"
                   "<p><b>Website:</b> <a href=\"%6\">%7</a></p>")
            .arg(QString::fromStdString(args_.app_name), QString::fromStdString(args_.app_version), __DATE__, __TIME__, QString::fromStdString(args_.email), QString::fromStdString(args_.website), QString::fromStdString(args_.website));
    }

private slots:
    void copyToClipboard()
    {
        QString plain_text = getVersionInfo(); // 获取版本信息

        QGuiApplication::clipboard()->setText(plain_text); // 仅设置剪贴板文本
    }
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr, Args &args = Args{});
    ~MainWindow();

protected:
    // 拖拽事件处理
    void dragEnterEvent(QDragEnterEvent *event) override;
    void dropEvent(QDropEvent *event) override;
    void mouseDoubleClickEvent(QMouseEvent *event) override;

private slots:
    // 加载图片
    void loadImage();
    // 执行图像分割
    void runSegmentation();
    // 保存分割结果
    void saveResults();
    void onModelSelected(QAction *action);
    void onDeviceSelected(QAction *action);
    void showAboutDialog();

private:
    void initUI();
    void initModels();
    cv::Mat QImageToCvMat(const QImage &image, bool clone_data = true);
    QImage cvMatToQImage(const cv::Mat &mat);
    void doSegmentation(QString file_path);
    void saveConfigWithQSettings();
    void loadConfigWithQSettings();

private:
    QStringList model_list_names_ = {"Large Model", "Small Model"};
    QStringList model_list_path_ = {"weights/BiRefNet_dynamic-general-epoch_174.onnx", "weights/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx"};
    QStringList devices_ = {"CPU"};
    int current_model_index_, device_id_;
    QVector<int> model_list_ids_;
    QString config_file_path_ = "config.ini";
    int win_h_ = 600, win_w_ = 1500;

    // UI组件
    QLabel *image_label_;      // 原始图片显示
    QLabel *result_label_;     // 分割结果显示
    QPushButton *load_button_; // 加载图片按钮
    QPushButton *save_button_; // 保存结果按钮

    // 数据
    QImage original_image_;      // 原始图片
    QPixmap original_pixmap_;    // 原始图片
    QPixmap result_pixmap_;      // 分割结果图片
    QString current_image_path_; // 当前图片路径

    std::shared_ptr<matting::Matter> matting_model_ = nullptr;
    std::shared_ptr<spdlog::logger> logger_;
    Args args_;
};

#endif // MAINWINDOW_H
