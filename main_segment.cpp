#include <QApplication>
#include <QTextCodec>

#include "app.h"
#include "interface/segment/mainwindow.h"


int main(int argc, char *argv[])
{
    std::cout << "start..." << std::endl;
    Args args;
    args.root_path = "./";
    args.app_name = "Easy Removal Background";
    args.app_version = "v1.0.0";
    args.email = "ming1451093037@gmail.com";
    args.website = "https://github.com/zhangming8/easyRemovalBackground";

    QApplication app(argc, argv);

    app.setWindowIcon(QIcon("./icon.png"));
    app.setApplicationName(args.app_name.c_str());
    app.setApplicationVersion(args.app_version.c_str());
    app.setOrganizationName("");

    MainWindow window(nullptr, args);
    window.show();
    int ret = app.exec();

    std::cout << "end..." << std::endl;
    return ret;
}
