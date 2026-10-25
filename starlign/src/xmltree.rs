//! A small XML DOM. Enough to mark up XHTML and rewrite an OPF package.

use anyhow::{bail, Context, Result};
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::{Reader, Writer};

#[derive(Clone, Debug)]
pub struct Document {
    pub nodes: Vec<Node>,
    pub root: usize,
    pub xml_decl: bool,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub parent: Option<usize>,
    pub data: NodeData,
}

#[derive(Clone, Debug)]
pub enum NodeData {
    Element {
        name: String,
        attrs: Vec<(String, String)>,
        children: Vec<usize>,
    },
    Text(String),
    Comment(String),
}

impl Document {
    pub fn parse(xml: &str) -> Result<Self> {
        let mut reader = Reader::from_str(xml);
        reader.config_mut().trim_text(false);
        let mut buf = Vec::new();
        let mut nodes: Vec<Node> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();
        let mut root = None;
        let mut xml_decl = false;

        loop {
            let event = reader.read_event_into(&mut buf).context("reading xml")?;
            match event {
                Event::Decl(_) => xml_decl = true,
                Event::Start(elem) => {
                    let idx = push_element(&mut nodes, &elem)?;
                    attach(&mut nodes, &mut stack, idx);
                    if root.is_none() {
                        root = Some(idx);
                    }
                    stack.push(idx);
                }
                Event::Empty(elem) => {
                    let idx = push_element(&mut nodes, &elem)?;
                    attach(&mut nodes, &mut stack, idx);
                    if root.is_none() {
                        root = Some(idx);
                    }
                }
                Event::End(_) => {
                    stack.pop();
                }
                Event::Text(text) => {
                    let value = text.unescape().context("xml text")?.into_owned();
                    if !value.is_empty() {
                        push_text(&mut nodes, &stack, value);
                    }
                }
                Event::CData(text) => {
                    let value = String::from_utf8_lossy(text.as_ref()).into_owned();
                    if !value.is_empty() {
                        push_text(&mut nodes, &stack, value);
                    }
                }
                Event::Comment(comment) => {
                    let value = String::from_utf8_lossy(comment.as_ref()).into_owned();
                    let idx = nodes.len();
                    nodes.push(Node {
                        parent: stack.last().copied(),
                        data: NodeData::Comment(value),
                    });
                    attach(&mut nodes, &stack, idx);
                }
                Event::Eof => break,
                _ => {}
            }
            buf.clear();
        }

        let Some(root) = root else {
            bail!("xml document has no root element");
        };
        Ok(Self {
            nodes,
            root,
            xml_decl,
        })
    }

    pub fn to_xml(&self) -> Result<String> {
        let mut writer = Writer::new(Vec::new());
        if self.xml_decl {
            writer.write_event(Event::Decl(BytesDecl::new("1.0", Some("utf-8"), None)))?;
            writer.write_event(Event::Text(BytesText::new("\n")))?;
        }
        self.write_node(&mut writer, self.root)?;
        let bytes = writer.into_inner();
        Ok(String::from_utf8(bytes)?)
    }

    fn write_node(&self, writer: &mut Writer<Vec<u8>>, idx: usize) -> Result<()> {
        match &self.nodes[idx].data {
            NodeData::Text(text) => {
                writer.write_event(Event::Text(BytesText::new(text)))?;
            }
            NodeData::Comment(text) => {
                writer.write_event(Event::Comment(BytesText::new(text)))?;
            }
            NodeData::Element {
                name,
                attrs,
                children,
            } => {
                let mut start = BytesStart::new(name.as_str());
                for (key, value) in attrs {
                    start.push_attribute((key.as_str(), value.as_str()));
                }
                if children.is_empty() {
                    writer.write_event(Event::Empty(start))?;
                } else {
                    writer.write_event(Event::Start(start))?;
                    for child in children {
                        self.write_node(writer, *child)?;
                    }
                    writer.write_event(Event::End(BytesEnd::new(name.as_str())))?;
                }
            }
        }
        Ok(())
    }

    pub fn create_element(&mut self, name: &str, attrs: Vec<(String, String)>) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            parent: None,
            data: NodeData::Element {
                name: name.to_string(),
                attrs,
                children: Vec::new(),
            },
        });
        idx
    }

    pub fn create_text(&mut self, text: String) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(Node {
            parent: None,
            data: NodeData::Text(text),
        });
        idx
    }

    pub fn children(&self, idx: usize) -> &[usize] {
        match &self.nodes[idx].data {
            NodeData::Element { children, .. } => children,
            _ => &[],
        }
    }

    pub fn set_children(&mut self, parent: usize, children: Vec<usize>) {
        for child in &children {
            self.nodes[*child].parent = Some(parent);
        }
        if let NodeData::Element { children: slot, .. } = &mut self.nodes[parent].data {
            *slot = children;
        }
    }

    pub fn take_children(&mut self, parent: usize) -> Vec<usize> {
        match &mut self.nodes[parent].data {
            NodeData::Element { children, .. } => std::mem::take(children),
            _ => Vec::new(),
        }
    }

    pub fn name(&self, idx: usize) -> &str {
        match &self.nodes[idx].data {
            NodeData::Element { name, .. } => name,
            _ => "",
        }
    }

    pub fn is_element(&self, idx: usize) -> bool {
        matches!(self.nodes[idx].data, NodeData::Element { .. })
    }

    pub fn text(&self, idx: usize) -> &str {
        match &self.nodes[idx].data {
            NodeData::Text(text) => text,
            _ => "",
        }
    }

    pub fn attr(&self, idx: usize, key: &str) -> Option<String> {
        let NodeData::Element { attrs, .. } = &self.nodes[idx].data else {
            return None;
        };
        attrs
            .iter()
            .find(|(name, _)| attr_is(name, key))
            .map(|(_, value)| value.clone())
    }

    pub fn set_attr(&mut self, idx: usize, key: &str, value: &str) {
        let NodeData::Element { attrs, .. } = &mut self.nodes[idx].data else {
            return;
        };
        if let Some((_, slot)) = attrs.iter_mut().find(|(name, _)| attr_is(name, key)) {
            *slot = value.to_string();
        } else {
            attrs.push((key.to_string(), value.to_string()));
        }
    }

    pub fn remove_attr(&mut self, idx: usize, key: &str) {
        let NodeData::Element { attrs, .. } = &mut self.nodes[idx].data else {
            return;
        };
        attrs.retain(|(name, _)| !attr_is(name, key));
    }

    pub fn elements_named(&self, local: &str) -> Vec<usize> {
        let mut out = Vec::new();
        let root = self.root;
        self.walk(root, &mut |doc, idx| {
            if doc.is_element(idx) && local_name(doc.name(idx)) == local {
                out.push(idx);
            }
        });
        out
    }

    /// Preorder walk. The visitor receives the document so it can read nodes
    /// without capturing `self` across the recursive borrow.
    pub fn walk(&self, idx: usize, visit: &mut dyn FnMut(&Document, usize)) {
        let mut stack = vec![idx];
        while let Some(current) = stack.pop() {
            visit(self, current);
            let children = self.children(current).to_vec();
            for child in children.into_iter().rev() {
                stack.push(child);
            }
        }
    }
}

fn push_element(nodes: &mut Vec<Node>, elem: &BytesStart) -> Result<usize> {
    let name = String::from_utf8_lossy(elem.name().as_ref()).into_owned();
    let mut attrs = Vec::new();
    for attr in elem.attributes() {
        let attr = attr.context("xml attribute")?;
        let key = String::from_utf8_lossy(attr.key.as_ref()).into_owned();
        let value = attr
            .unescape_value()
            .context("xml attribute value")?
            .into_owned();
        attrs.push((key, value));
    }
    let idx = nodes.len();
    nodes.push(Node {
        parent: None,
        data: NodeData::Element {
            name,
            attrs,
            children: Vec::new(),
        },
    });
    Ok(idx)
}

fn attach(nodes: &mut [Node], stack: &[usize], idx: usize) {
    if let Some(&parent) = stack.last() {
        nodes[idx].parent = Some(parent);
        if let NodeData::Element { children, .. } = &mut nodes[parent].data {
            children.push(idx);
        }
    }
}

fn push_text(nodes: &mut Vec<Node>, stack: &[usize], value: String) {
    let Some(&parent) = stack.last() else {
        return;
    };
    if let NodeData::Element { children, .. } = &nodes[parent].data {
        if let Some(&last) = children.last() {
            if let NodeData::Text(existing) = &mut nodes[last].data {
                existing.push_str(&value);
                return;
            }
        }
    }
    let idx = nodes.len();
    nodes.push(Node {
        parent: Some(parent),
        data: NodeData::Text(value),
    });
    if let NodeData::Element { children, .. } = &mut nodes[parent].data {
        children.push(idx);
    }
}

pub fn local_name(name: &str) -> &str {
    name.rsplit(':').next().unwrap_or(name)
}

fn attr_is(name: &str, key: &str) -> bool {
    name == key || local_name(name) == local_name(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_keeps_text_and_attributes() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body><p class="a">Bill &amp; Bob<br/>end</p></body></html>"#;
        let doc = Document::parse(xml).unwrap();
        let out = doc.to_xml().unwrap();
        assert!(out.contains("Bill &amp; Bob"), "{out}");
        assert!(out.contains("class=\"a\""), "{out}");
        assert!(out.contains("<br/>") || out.contains("<br />"), "{out}");
        assert!(out.contains("end"), "{out}");
    }
}
