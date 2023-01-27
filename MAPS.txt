# The Mercator projected raster maps were created as follows, where ne_10m_admin_0_scale_rank_minor_islands.shp is a shapefile from https://www.naturalearthdata.com/

# we first convert to Plate-Carree since gdal_rasterize won't reproject

gdal_rasterize -burn 255 -where "sr_sov_a3 = 'US1'" -ts 32768 32768 -ot Byte ne_10m_admin_0_scale_rank_minor_islands.shp -of bmp -te -180 -90 180 90 usa-raster-plate-carree.bmp

# we now project for every slippy tile zoomlevel: since 32768 x 32768 is equivalent to 128 x 128 tiles of 256 x 256 piles each, and 2**7 == 128, usa-merc-7.bmp tiles zoom level 7 (and so on down the line)

# I originally tried to use Imagemagick's convert with `-resize 50%` to create the lower resolution maps, but it kept grayscaling my monochrome images, even with options like `-dither None`, `-type bilevel`, `-depth 1` , `-colors 2` and so on

perl -le 'for $i (0..7) {$size = 256*2**$i; print "gdalwarp -te -20037508.342789244 -20037508.342789244 20037508.342789244 20037508.342789244 -s_srs EPSG:4326 -t_srs EPSG:3857 -of BMP -ts $size $size usa-raster-plate-carree.bmp usa-merc-$i.bmp"}'

# piping above to tcsh runs the commands

# NOTE: the bmp files above are too large for git and contain unnecessary information, so we convert to PNG (which, when done like this, preserves monochromacity, even though `-resize 50%` doesnt)

ls usa-merc-*.bmp | perl -nle 's/\.bmp//; print "convert $_.bmp $_.png"'

# and pipe above to tcsh to get PNG files in this git


