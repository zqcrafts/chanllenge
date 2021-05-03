read -r -p "Are You Sure to **UPLOAD WEIGHT**? [Y/n] " input

case $input in
    [yY][eE][sS]|[yY])
		echo "Yes"
        rm work_dirs.zip
        zip -r work_dirs.zip work_dirs
        HDFS_PREFIX=hdfs://haruna/home/byte_labcv_default/user/chenzehui.123
        hdfs dfs -rm -r ${HDFS_PREFIX}/work_dirs.zip
        hadoop fs -put -f work_dirs.zip ${HDFS_PREFIX}/
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