# list of soverignties

if $argv[1] == 1 then

ogrinfo -sql "SELECT DISTINCT sr_sov_a3 FROM ne_10m_admin_0_scale_rank_minor_islands" ../ne_10m_admin_0_scale_rank_minor_islands.shp | fgrep 'sr_sov_a3' | perl -anle 'print $F[3]' | sort -u | egrep -v '^$' | tee countries.txt

endif

# create highres map for each soverignty

if $argv[1] == 2 then

perl -nle 'print(qq%gdal_rasterize -burn 255 -where "sr_sov_a3 = \47$_\47" -ts 43200 21600 -ot Byte ../ne_10m_admin_0_scale_rank_minor_islands.shp -of bmp -te -180 -90 180 90 $_-raster-43200.bmp%)' countries.txt

# perl -nle 'print(qq%gdal_rasterize -burn 255 -where "sr_sov_a3 = \47$_\47" -ts 43200 21600 -ot Byte ne_10m_admin_0_scale_rank_minor_islands.shp -of bmp -te -180 -90 180 90 $_-raster-43200.bmp; convert $_-raster-43200.bmp $_-ra%)' countries.txt

endif


# nicer list of countries?

if $argv[1] == 3 then

ogrinfo -sql "SELECT CONCAT(sr_sov_a3, ' ', sr_subunit) FROM ne_10m_admin_0_scale_rank_minor_islands" -geom=NO ../ne_10m_admin_0_scale_rank_minor_islands.shp | perl -anle 'if (s%\s*CONCAT_sr_sov_a3 \(String\) = %%) {print $_}' | sort -u

endif

if $argv[1] == 4 then

perl -nle 'print(qq%gdal_rasterize -burn 255 -where "sr_sov_a3 = \47$_\47" -ts 16384 8192 -ot Byte ../ne_10m_admin_0_scale_rank_minor_islands.shp -of bmp -te -180 -90 180 90 $_-raster-16384.bmp%)' countries.txt

endif


if $argv[1] == 5 then

perl -nle 'print(qq%gdal_rasterize -burn 255 -where "sr_sov_a3 = \47$_\47" -ts 2048 1024 -ot Byte ../ne_10m_admin_0_scale_rank_minor_islands.shp -of bmp -te -180 -90 180 90 $_-raster-2048.bmp%)' countries.txt

endif


