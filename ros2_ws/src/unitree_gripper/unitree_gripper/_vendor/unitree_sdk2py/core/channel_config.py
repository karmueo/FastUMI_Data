"""为夹爪 DDS 通道生成指定网卡或自动选网卡的配置。"""

# CycloneDDS 0.10.2 在本机设置 Tracing/Verbosity 时可能触发原生崩溃。
ChannelConfigHasInterface = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface name="$__IF_NAME__$" priority="default" multicast="default"/>
                </Interfaces>
            </General>
            <Tracing><OutputFile>/tmp/cdds.LOG</OutputFile></Tracing>
        </Domain>
    </CycloneDDS>'''

ChannelConfigAutoDetermine = '''<?xml version="1.0" encoding="UTF-8" ?>
    <CycloneDDS>
        <Domain Id="any">
            <General>
                <Interfaces>
                    <NetworkInterface autodetermine=\"true\" priority=\"default\" multicast=\"default\" />
                </Interfaces>
            </General>
            <Tracing><OutputFile>/tmp/cdds.LOG</OutputFile></Tracing>
        </Domain>
    </CycloneDDS>'''
