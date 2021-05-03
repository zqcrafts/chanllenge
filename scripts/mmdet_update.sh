read -r -p "Are You Sure to **UPDATE**? [Y/n] " input

case $input in
    [yY][eE][sS]|[yY])
		echo "Yes"
        cd ..
        rm -rf mmdetection
        # rm -rf mmdetection.zip
        rm mmdetection.tar.gz
        HDFS_PREFIX=hdfs://haruna/home/byte_labcv_default/user/chenzehui.123/
        hdfs dfs -copyToLocal $HDFS_PREFIX/mmdetection.tar.gz ./
        # unzip -q mmdetection.zip
        tar -xvf mmdetection.tar.gz
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
