read -r -p "Are You Sure to **UPDATE WEIGHT**? [Y/n] " input

case $input in
    [yY][eE][sS]|[yY])
		echo "Yes"
        rm work_dirs.zip
        HDFS_PREFIX=hdfs://haruna/home/byte_labcv_default/user/chenzehui.123/
        hdfs dfs -copyToLocal $HDFS_PREFIX/work_dirs.zip ./
        unzip -q work_dirs.zip
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
