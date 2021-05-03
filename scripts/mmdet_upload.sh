read -r -p "Are You Sure to **UPLOAD**? [Y/n] " input

case $input in
    [yY][eE][sS]|[yY])
		echo "Yes"
        cd ..
        rm -rf mmdetection/work_dirs
        rm mmdetection/work_dirs.zip
        rm -rf mmdetection/widerface_evaluate/widerface_txt
        rm mmdetection/data
        rm mmdetection.tar.gz
        rm mmdetection.zip
        # zip -r mmdetection.zip mmdetection
        tar -czvf mmdetection.tar.gz mmdetection
        HDFS_PREFIX=hdfs://haruna/home/byte_labcv_default/user/chenzehui.123
        # hdfs dfs -rm -r ${HDFS_PREFIX}/mmdetection.zip/
        hdfs dfs -rm -r ${HDFS_PREFIX}/mmdetection.tar.gz
        hadoop fs -put -f mmdetection.tar.gz ${HDFS_PREFIX}/
        ln -s /dev/shm/chenzeh/data mmdetection/data
        cd mmdetection
		;;

    [nN][oO]|[nN])
		echo "No"
        return
       	;;

    *)
		echo "Invalid input..."
        return
		;;
esac
echo "abc"